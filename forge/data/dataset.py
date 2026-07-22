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
import math
import os
import shutil
import zipfile

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


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
    and returns zero, so the ordinary exact-final selection path remains usable.
    """
    copied: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []
    chosen: list[tuple[str, str]] = []
    original_digests: dict[tuple[str, str], str] = {}
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


def _strict_reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    os.makedirs(path)


def _flat_pairs(directory: str) -> list[tuple[str, str]]:
    if not os.path.isdir(directory):
        return []
    out: list[tuple[str, str]] = []
    seen_stems: set[str] = set()
    for name in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in _IMAGE_EXTS:
            continue
        stem_key = stem.casefold()
        if stem_key in seen_stems:
            raise RuntimeError(
                f"ambiguous image stem in flat dataset: {stem!r}"
            )
        image_path = os.path.join(directory, name)
        caption_path = os.path.join(directory, stem + ".txt")
        if os.path.isfile(image_path) and os.path.isfile(caption_path):
            seen_stems.add(stem_key)
            out.append((image_path, caption_path))
    return out


def _pair_digest(image_path: str, caption_path: str) -> str:
    digest = hashlib.sha256()
    for path in (image_path, caption_path):
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


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
