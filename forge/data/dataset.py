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
