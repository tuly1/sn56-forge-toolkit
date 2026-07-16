"""Perceptual dedup before captioning. dHash AND mean-RGB colour signature,
union-find at Hamming<=2 AND colour-dist<=6.0. Pure PIL+numpy. Never raises.

Thresholds are deliberately NEAR-EXACT-ONLY: tournament datasets are 10-50
pairs and the evaluator holds out ceil(0.1*N) images for reconstruction
scoring — deleting a visually-distinct subject (false positive) costs held-out
coverage, while keeping a true duplicate costs almost nothing. Low-texture /
uniform-background sets false-positive badly at looser thresholds."""
from __future__ import annotations

import math
import os

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_SIDECARS = (".txt", ".npz", ".npy", ".caption")
_HAMMING_MAX = 2
_COLOUR_MAX = 6.0
_MIN_N = 15                 # sets smaller than this are never touched
_KEEP_FRAC = 0.6           # never thin below max(15, ceil(0.6*N))


def dedup_dataset(image_dir: str) -> int:
    """Remove near-duplicate images (and their sidecars) in image_dir.
    Returns count removed. Any error -> 0 removed (safe no-op)."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return 0
    try:
        names = sorted(e for e in os.listdir(image_dir)
                       if e.lower().endswith(_IMAGE_EXTS))
        n = len(names)
        if n < _MIN_N:
            return 0
        keep_floor = max(_MIN_N, math.ceil(_KEEP_FRAC * n))

        dhash: list[int] = []
        colour: list[tuple[float, float, float]] = []
        valid: list[str] = []
        for name in names:
            try:
                with Image.open(os.path.join(image_dir, name)) as im:
                    rgb = im.convert("RGB")
                    # dHash: grayscale 9x8, compare adjacent columns -> 64 bits
                    g = np.asarray(rgb.resize((9, 8)).convert("L"), dtype=np.int16)
                    bits = (g[:, 1:] > g[:, :-1]).flatten()
                    h = 0
                    for b in bits:
                        h = (h << 1) | int(b)
                    # colour signature: mean RGB over a downscaled copy
                    small = np.asarray(rgb.resize((32, 32)), dtype=np.float32)
                    c = tuple(small.reshape(-1, 3).mean(axis=0))
            except Exception:
                continue
            dhash.append(h)
            colour.append(c)
            valid.append(name)

        m = len(valid)
        if m < _MIN_N:
            return 0

        # union-find grouping: edge iff Hamming(dhash)<=6 AND colourdist<=14 (BOTH)
        parent = list(range(m))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(m):
            for j in range(i + 1, m):
                if bin(dhash[i] ^ dhash[j]).count("1") > _HAMMING_MAX:
                    continue
                dc = ((colour[i][0] - colour[j][0]) ** 2 +
                      (colour[i][1] - colour[j][1]) ** 2 +
                      (colour[i][2] - colour[j][2]) ** 2) ** 0.5
                if dc <= _COLOUR_MAX:
                    union(i, j)

        # clusters, largest first; drop all-but-one per cluster, respect floor
        clusters: dict[int, list[int]] = {}
        for i in range(m):
            clusters.setdefault(find(i), []).append(i)
        groups = sorted(clusters.values(), key=len, reverse=True)

        def has_caption(idx: int) -> bool:
            stem = os.path.splitext(valid[idx])[0]
            p = os.path.join(image_dir, stem + ".txt")
            try:
                return os.path.isfile(p) and os.path.getsize(p) > 0
            except Exception:
                return False

        remaining = m
        to_remove: list[int] = []
        for grp in groups:
            if len(grp) <= 1:
                continue
            # Representative = first CAPTIONED member (else first member), so a
            # cluster never keeps an uncaptioned copy while deleting the only
            # caption for that subject.
            rep = next((i for i in grp if has_caption(i)), grp[0])
            for idx in grp:
                if idx == rep:
                    continue
                if remaining <= keep_floor:
                    break
                to_remove.append(idx)
                remaining -= 1
            if remaining <= keep_floor:
                break

        removed = 0
        for idx in to_remove:
            stem = os.path.splitext(valid[idx])[0]
            for path in ([os.path.join(image_dir, valid[idx])] +
                         [os.path.join(image_dir, stem + s) for s in _SIDECARS]):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass
            removed += 1
        return removed
    except Exception:
        return 0
