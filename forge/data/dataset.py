"""Unpack the pre-staged image dataset into ai-toolkit's expected layout.

The validator stages a zip at ``/cache/datasets/{task_id}_tourn.zip`` (read-only
cache, no internet). Inside is a folder of paired files: each image
(.png/.jpg/.jpeg/.webp) beside a same-basename ``.txt`` caption. ai-toolkit reads
a FLAT directory: image files + same-basename ``.txt`` captions in one folder
(NOT kohya's ``{repeats}_concept`` structure, no ``img/`` parent).

Critical vs the legacy SDXL/kohya dataset:
  * Captions are copied BYTE-EXACT (``shutil.copyfile`` — never decode/encode).
    This preserves ideogram4's compact-JSON captions verbatim.
  * NO trigger-prepend, NO keep_tokens, NO concept folder. ai-toolkit injects the
    trigger via config ``trigger_word``, so a missing caption becomes an EMPTY
    ``.txt`` (not a bare trigger — that would double-inject and break the
    byte-exact invariant).

Pure stdlib — no torch — so it is cheap to unit-test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import zipfile

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_HOLDOUT_RECEIPT_FILE = ".forge_holdout_reservation.json"
_HOLDOUT_RECEIPT_SCHEMA = 1


def prepare_aitoolkit_dataset(
    zip_path: str,
    *,
    images_dir: str,
    trigger_word: str | None = None,
) -> tuple[str, int]:
    """Unzip → descend → dedup → flat byte-exact copy into ``images_dir``.

    Returns (images_dir, num_pairs). Raises FileNotFoundError/RuntimeError only
    (the caller in ``aitoolkit.run`` funnels to fallback). ``trigger_word`` is
    accepted for signature symmetry but INTENTIONALLY UNUSED — ai-toolkit injects
    it at train time via the config.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"dataset zip not found at {zip_path!r}")

    work = images_dir.rstrip("/") + "__extract"
    flat = images_dir.rstrip("/") + "__flat"
    _reset_dir(work)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work)

    # Flatten EVERY subdirectory into one staging dir (mirrors the validator's own
    # staging: rglob over all subdirs, collision-rename on basename). A zip that
    # splits images across sibling/per-concept folders must NOT be truncated to a
    # single subdir — that would shrink N and mis-scale the step budget.
    src = _collect_flat(work, flat)
    if src is None:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"no images found inside {zip_path!r}")

    # Perceptual dedup BEFORE the copy loop so near-duplicates and their sidecars
    # are gone before counting. Runs on the fully-collected flat set. Never raises
    # → 0 on error (INV-1).
    try:
        from forge.data import dedup

        num_removed = dedup.dedup_dataset(src)
    except Exception:
        num_removed = 0

    _reset_dir(images_dir)  # FLAT target
    pairs = 0
    for name in sorted(os.listdir(src)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in _IMAGE_EXTS:
            continue
        shutil.copy2(os.path.join(src, name), os.path.join(images_dir, name))
        src_cap = os.path.join(src, stem + ".txt")
        dst_cap = os.path.join(images_dir, stem + ".txt")
        if os.path.isfile(src_cap):
            shutil.copyfile(src_cap, dst_cap)  # BYTE-EXACT, no decode/encode
        else:
            open(dst_cap, "wb").close()  # empty caption; ai-toolkit injects trigger
        pairs += 1

    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(flat, ignore_errors=True)
    if pairs == 0:
        raise RuntimeError(f"no usable image/caption pairs in {zip_path!r}")

    try:
        from forge import telemetry

        telemetry.event("dedup", removed=num_removed, kept=pairs)
    except Exception:
        pass
    return images_dir, pairs


def reserve_holdout(
    images_dir: str,
    *,
    holdout_dir: str,
    min_training_pairs: int = 3,
    max_holdout_pairs: int = 4,
) -> int:
    """Move a deterministic post-dedup subset out of the training directory.

    Selection must never score examples used for optimization.  We rank pairs
    by a content digest (image bytes plus caption bytes), not by caller-provided
    filenames, and reserve roughly ten percent: one pair at the current minimum
    tournament size and at most four for larger tasks.

    This helper is deliberately fail-open for *training* and fail-closed for
    *selection*.  It copies every chosen pair before deleting any source.  On
    any error it restores the original training directory, clears the holdout,
    and returns zero only after proving byte-exact rollback. An ambiguous
    rollback raises so the handler falls back without training on changed data.
    """
    copied: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []
    chosen: list[tuple[str, str]] = []
    original_digests: dict[tuple[str, str], str] = {}
    original_records: list[dict[str, str]] = []
    try:
        pairs = _flat_pairs(images_dir)
        min_training_pairs = max(1, int(min_training_pairs))
        max_holdout_pairs = max(0, int(max_holdout_pairs))
        if len(pairs) <= min_training_pairs or max_holdout_pairs == 0:
            _strict_reset_dir(holdout_dir)
            _holdout_event("holdout_unavailable", pairs=len(pairs))
            return 0

        count = min(
            max_holdout_pairs,
            max(1, int(math.ceil(len(pairs) * 0.10))),
            len(pairs) - min_training_pairs,
        )
        ranked = sorted(
            pairs,
            key=lambda pair: (_pair_digest(*pair), os.path.basename(pair[0])),
        )
        chosen = ranked[:count]
        original_records = _pair_records(pairs)
        original_digests = {
            pair: _pair_digest(*pair)
            for pair in chosen
        }
        _strict_reset_dir(holdout_dir)

        # Copy the complete set first.  A partial copy cannot shrink training.
        for image_path, caption_path in chosen:
            holdout_image = os.path.join(
                holdout_dir, os.path.basename(image_path)
            )
            holdout_caption = os.path.join(
                holdout_dir, os.path.basename(caption_path)
            )
            shutil.copy2(
                image_path,
                holdout_image,
            )
            shutil.copyfile(caption_path, holdout_caption)
            copied.extend(
                ((image_path, holdout_image), (caption_path, holdout_caption))
            )

        # Prove every reserved copy still equals its pre-split source, then
        # durably anchor both the reserved subset and the complete pre-split
        # dataset before deleting a single training input.  A later attestation
        # failure can only resume training if this exact dataset is restored.
        for image_path, caption_path in chosen:
            holdout_pair = (
                os.path.join(holdout_dir, os.path.basename(image_path)),
                os.path.join(holdout_dir, os.path.basename(caption_path)),
            )
            if _pair_digest(*holdout_pair) != original_digests[
                (image_path, caption_path)
            ]:
                raise RuntimeError("reserved holdout copy changed pair bytes")
        _write_holdout_receipt(
            holdout_dir,
            original_records=original_records,
            reserved_records=_pair_records(
                [
                    (
                        os.path.join(holdout_dir, os.path.basename(image_path)),
                        os.path.join(holdout_dir, os.path.basename(caption_path)),
                    )
                    for image_path, caption_path in chosen
                ]
            ),
        )

        for source, holdout_copy in copied:
            os.remove(source)
            removed.append((source, holdout_copy))

        _holdout_event(
            "holdout_reserved",
            heldout=count,
            training_pairs=len(pairs) - count,
        )
        return count
    except BaseException as exc:
        # Restore only current-run sources that we ourselves removed. Never
        # trust or import arbitrary files found in a reused holdout directory.
        for source, holdout_copy in removed:
            try:
                if not os.path.exists(source) and os.path.isfile(holdout_copy):
                    shutil.copy2(holdout_copy, source)
            except Exception:
                pass
        shutil.rmtree(holdout_dir, ignore_errors=True)
        try:
            os.makedirs(holdout_dir, exist_ok=True)
        except Exception:
            pass
        rollback_errors = []
        for pair in chosen:
            try:
                if (
                    not all(os.path.isfile(path) for path in pair)
                    or _pair_digest(*pair) != original_digests[pair]
                ):
                    rollback_errors.append(os.path.basename(pair[0]))
            except Exception:
                rollback_errors.append(os.path.basename(pair[0]))
        if original_records:
            try:
                if _pair_records(_flat_pairs(images_dir)) != original_records:
                    rollback_errors.append("dataset-inventory")
            except Exception:
                rollback_errors.append("dataset-inventory")
        _holdout_event(
            "holdout_reservation_failed", error=f"{type(exc).__name__}: {exc}"
        )
        if rollback_errors:
            _holdout_event(
                "holdout_rollback_incomplete",
                pairs=sorted(set(rollback_errors)),
            )
            raise RuntimeError(
                "holdout reservation failed and training inputs could not be "
                f"restored byte-exactly: {sorted(set(rollback_errors))}"
            ) from exc
        return 0


def restore_reserved_holdout(images_dir: str, *, holdout_dir: str) -> int:
    """Restore only the exact pre-split dataset bound by the durable receipt."""
    restored: list[tuple[str, str]] = []
    try:
        receipt = _read_holdout_receipt(holdout_dir)
        pairs = _flat_pairs(
            holdout_dir,
            allowed_files=frozenset({_HOLDOUT_RECEIPT_FILE}),
        )
        if _pair_records(pairs) != receipt["reserved_records"]:
            raise RuntimeError("reserved holdout bytes differ from the receipt")
        for image, caption in pairs:
            destinations = (
                os.path.join(images_dir, os.path.basename(image)),
                os.path.join(images_dir, os.path.basename(caption)),
            )
            if any(os.path.lexists(path) for path in destinations):
                raise RuntimeError("reserved holdout restore would overwrite training")
        for image, caption in pairs:
            destination_image = os.path.join(images_dir, os.path.basename(image))
            destination_caption = os.path.join(images_dir, os.path.basename(caption))
            expected = _pair_digest(image, caption)
            shutil.copy2(image, destination_image)
            shutil.copyfile(caption, destination_caption)
            if _pair_digest(destination_image, destination_caption) != expected:
                raise RuntimeError("reserved holdout restore changed pair bytes")
            restored.append((destination_image, destination_caption))
        if _pair_records(_flat_pairs(images_dir)) != receipt["original_records"]:
            raise RuntimeError(
                "restored training dataset differs from the pre-split receipt"
            )
        shutil.rmtree(holdout_dir)
        os.makedirs(holdout_dir)
        return len(restored)
    except BaseException:
        # The caller must abort the handler rather than train against an
        # ambiguous reconstruction. Any copied files remain available for
        # forensic recovery, but are never certified by a successful return.
        raise


def validate_reserved_split(
    images_dir: str,
    *,
    holdout_dir: str,
    split_identity: dict[str, object] | None = None,
) -> None:
    """Prove the live split is the exact partition captured before reservation."""
    receipt = _read_holdout_receipt(holdout_dir)
    training_records = _pair_records(_flat_pairs(images_dir))
    reserved_records = _pair_records(
        _flat_pairs(
            holdout_dir,
            allowed_files=frozenset({_HOLDOUT_RECEIPT_FILE}),
        )
    )
    if reserved_records != receipt["reserved_records"]:
        raise RuntimeError("reserved holdout bytes differ from the receipt")
    current_union = sorted(
        training_records + reserved_records,
        key=lambda record: record["image"].casefold(),
    )
    if current_union != receipt["original_records"]:
        raise RuntimeError("live dataset split differs from the pre-split receipt")
    if split_identity is not None:
        expected_training = _split_identity_records(training_records)
        expected_holdout = _split_identity_records(reserved_records)
        if (
            split_identity.get("training") != expected_training
            or split_identity.get("holdout") != expected_holdout
            or split_identity.get("training_pairs") != len(expected_training)
            or split_identity.get("holdout_pairs") != len(expected_holdout)
            or split_identity.get("total_pairs")
            != len(expected_training) + len(expected_holdout)
        ):
            raise RuntimeError(
                "dataset split identity differs from the reservation receipt"
            )


def count_flat_pairs(images_dir: str) -> int:
    """Return the strict complete-pair count used by holdout reservation."""
    return len(_flat_pairs(images_dir))


def strict_flat_pairs(
    directory: str,
    *,
    allowed_files: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """Inventory one flat dataset root, rejecting every unbound direct child."""
    return _flat_pairs(directory, allowed_files=allowed_files)


def _write_holdout_receipt(
    holdout_dir: str,
    *,
    original_records: list[dict[str, str]],
    reserved_records: list[dict[str, str]],
) -> None:
    receipt = {
        "schema": _HOLDOUT_RECEIPT_SCHEMA,
        "identity": "forge-pre-split-flat-dataset-v1",
        "original_records": original_records,
        "reserved_records": reserved_records,
    }
    path = os.path.join(holdout_dir, _HOLDOUT_RECEIPT_FILE)
    temp = path + ".tmp"
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(holdout_dir)
    except BaseException:
        try:
            if os.path.exists(temp):
                os.remove(temp)
        except Exception:
            pass
        raise


def _read_holdout_receipt(holdout_dir: str) -> dict[str, object]:
    path = os.path.join(holdout_dir, _HOLDOUT_RECEIPT_FILE)
    if os.path.islink(holdout_dir) or os.path.islink(path):
        raise RuntimeError("holdout reservation receipt path is unsafe")
    try:
        with open(path, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except Exception as exc:
        raise RuntimeError("holdout reservation receipt is unavailable") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != _HOLDOUT_RECEIPT_SCHEMA
        or receipt.get("identity") != "forge-pre-split-flat-dataset-v1"
    ):
        raise RuntimeError("holdout reservation receipt is invalid")
    original = _validate_receipt_records(receipt.get("original_records"))
    reserved = _validate_receipt_records(receipt.get("reserved_records"))
    if not reserved or any(record not in original for record in reserved):
        raise RuntimeError("holdout reservation receipt has an invalid subset")
    return {"original_records": original, "reserved_records": reserved}


def _validate_receipt_records(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("holdout reservation receipt has no records")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    expected_keys = {
        "image",
        "caption",
        "image_sha256",
        "caption_sha256",
        "pair_sha256",
    }
    for value_record in value:
        if not isinstance(value_record, dict) or set(value_record) != expected_keys:
            raise RuntimeError("holdout reservation receipt record is invalid")
        if not all(isinstance(value_record[key], str) for key in expected_keys):
            raise RuntimeError("holdout reservation receipt record is invalid")
        record = {key: str(value_record[key]) for key in expected_keys}
        image, caption = record["image"], record["caption"]
        stem, extension = os.path.splitext(image)
        if (
            not stem
            or extension.lower() not in _IMAGE_EXTS
            or os.path.basename(image) != image
            or os.path.basename(caption) != caption
            or "/" in image + caption
            or "\\" in image + caption
            or caption != stem + ".txt"
            or image.casefold() in seen
            or any(
                not isinstance(record[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None
                for key in ("image_sha256", "caption_sha256", "pair_sha256")
            )
        ):
            raise RuntimeError("holdout reservation receipt record is unsafe")
        seen.add(image.casefold())
        records.append(record)
    return records


def _strict_reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    os.makedirs(path)


def _flat_pairs(
    directory: str,
    *,
    allowed_files: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    if os.path.islink(directory):
        raise RuntimeError(f"flat dataset root is unsafe: {directory!r}")
    if not os.path.isdir(directory):
        return []
    images: list[tuple[str, str]] = []
    captions: set[str] = set()
    seen_stems: set[str] = set()
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.islink(path):
            raise RuntimeError(f"flat dataset contains a symlink: {name!r}")
        if name in allowed_files:
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"allowed dataset sidecar is not a regular file: {name!r}"
                )
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".txt":
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"dataset caption is not a regular file: {name!r}"
                )
            captions.add(name)
            continue
        if ext.lower() not in _IMAGE_EXTS or not os.path.isfile(path):
            raise RuntimeError(f"flat dataset contains an unbound entry: {name!r}")
        stem_key = stem.casefold()
        if stem_key in seen_stems:
            raise RuntimeError(
                f"ambiguous image stem in flat dataset: {stem!r}"
            )
        seen_stems.add(stem_key)
        images.append((name, stem + ".txt"))
    expected_captions = {caption for _image, caption in images}
    if captions != expected_captions:
        unexpected = sorted(captions - expected_captions)
        missing = sorted(expected_captions - captions)
        raise RuntimeError(
            "flat dataset caption inventory is not one-to-one: "
            f"unexpected={unexpected}, missing={missing}"
        )
    return [
        (os.path.join(directory, image), os.path.join(directory, caption))
        for image, caption in images
    ]


def _pair_digest(image_path: str, caption_path: str) -> str:
    return _component_pair_digest(
        _file_sha256(image_path),
        _file_sha256(caption_path),
    )


def _component_pair_digest(image_sha256: str, caption_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"forge-image-caption-pair-v2\0")
    digest.update(bytes.fromhex(image_sha256))
    digest.update(bytes.fromhex(caption_sha256))
    return digest.hexdigest()


def _pair_records(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for image_path, caption_path in pairs:
        if os.path.islink(image_path) or os.path.islink(caption_path):
            raise RuntimeError("dataset pair path is unsafe")
        image_sha256 = _file_sha256(image_path)
        caption_sha256 = _file_sha256(caption_path)
        records.append(
            {
                "image": os.path.basename(image_path),
                "caption": os.path.basename(caption_path),
                "image_sha256": image_sha256,
                "caption_sha256": caption_sha256,
                "pair_sha256": _component_pair_digest(
                    image_sha256,
                    caption_sha256,
                ),
            }
        )
    return sorted(records, key=lambda record: record["image"].casefold())


def _split_identity_records(
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for record in sorted(records, key=lambda value: value["image"]):
        sample = hashlib.sha256()
        sample.update(b"forge-image-caption-v1\0")
        sample.update(bytes.fromhex(record["image_sha256"]))
        sample.update(bytes.fromhex(record["caption_sha256"]))
        out.append(
            {
                "image_sha256": record["image_sha256"],
                "caption_sha256": record["caption_sha256"],
                "sample_sha256": sample.hexdigest(),
            }
        )
    return out


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _holdout_event(name: str, **values) -> None:
    try:
        from forge import telemetry

        telemetry.event(name, **values)
    except Exception:
        pass


def _collect_flat(root: str, dest: str) -> str | None:
    """Walk EVERY subdirectory of ``root`` and copy each image plus its
    same-basename ``.txt`` into a single flat ``dest`` dir. On basename collision,
    prefix with the parent-folder name (``{parent}_{name}``), exactly as the
    validator's staging does, so images split across per-concept folders are all
    kept rather than truncated to one folder. Returns ``dest`` if it holds at
    least one image, else ``None``. Captions are copied byte-exact and re-paired
    to the (possibly renamed) image stem so pairing survives the rename.
    """
    _reset_dir(dest)
    found = False
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in _IMAGE_EXTS:
                continue
            new_name = _unique_name(dest, fn, os.path.basename(dirpath.rstrip("/")))
            new_stem = os.path.splitext(new_name)[0]
            shutil.copy2(
                os.path.join(dirpath, fn), os.path.join(dest, new_name)
            )
            cap = _find_caption(dirpath, files, stem)
            if cap:
                shutil.copyfile(cap, os.path.join(dest, new_stem + ".txt"))
            found = True
    return dest if found else None


def _find_caption(dirpath: str, files: list[str], stem: str) -> str | None:
    """Sidecar caption for ``stem``: same stem plus ``.txt`` with the extension
    matched case-insensitively, so IMG0.TXT pairs with IMG0.JPG on Linux the
    same way it appears to on a case-insensitive dev filesystem."""
    for f in files:
        s, e = os.path.splitext(f)
        if s == stem and e.lower() == ".txt":
            return os.path.join(dirpath, f)
    return None


def _unique_name(dest: str, name: str, parent: str) -> str:
    """A destination filename whose STEM is not yet claimed in ``dest``: first
    the bare name, then ``{parent}_{name}``, then numeric suffixes. Keying
    collisions on the stem (not the full filename) keeps ``a.jpg`` and ``a.png``
    from both landing bare and cross-wiring the shared ``a.txt`` caption."""
    stem, ext = os.path.splitext(name)
    if not _stem_taken(dest, stem):
        return name
    p_stem = f"{parent}_{stem}" if parent else stem
    if p_stem != stem and not _stem_taken(dest, p_stem):
        return f"{p_stem}{ext}"
    i = 1
    while True:
        cand = f"{p_stem}_{i}"
        if not _stem_taken(dest, cand):
            return f"{cand}{ext}"
        i += 1


def _stem_taken(dest: str, stem: str) -> bool:
    low = stem.lower()
    for f in os.listdir(dest):
        s, e = os.path.splitext(f)
        if e.lower() in _IMAGE_EXTS and s.lower() == low:
            return True
    return False


def _reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
