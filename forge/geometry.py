"""Evaluator-aligned training geometry — the `resolution` policy.

THE DEFECT THIS ADDRESSES
=========================
All five of our templates inherit ``resolution: [512, 768, 1024]`` byte-identical
from the upstream ai-toolkit template.  ai-toolkit's
``preprocess_dataset_raw_config`` (config_modules.py:1044-1062) forks a
resolution LIST into three INDEPENDENT dataset copies of the same folder, so a
task with N pairs materialises 3N training samples — and each copy buckets on a
TOTAL-PIXEL cap (``toolkit/buckets.py:get_bucket_for_image_size``: total pixels
<= resolution**2, aspect preserved, BICUBIC).

The evaluator does something structurally different: it scales the LONG EDGE to
1024 with LANCZOS, floors both dims to a multiple of 16, and centre-crops
(``evaluator/image_io.py:adjust_image_size``, replicated exactly below).

Measured over the fourteen real Aug-3 task shapes (473 source images, so 1419
materialised samples): **270 of 1419 = 19.0%** land at the exact geometry the
evaluator scores.  flux is at 0.0%.  Wide sources (1408x768, 1376x768,
1195x896 — six of fourteen tasks) match at NO resolution copy, because the
``@1024`` bucket keeps a ~1400 px long edge where the evaluator scores at 1024:
a 1.8-1.9x pixel-count difference.

WHAT THIS MODULE DOES
=====================
Given the ACTUAL source dimensions of the prepared dataset, it searches for the
single scalar ``resolution`` R whose ai-toolkit bucket reproduces the
evaluator's scored size for as many images as possible, and emits that as ONE
resolution value instead of the three-copy list.  Replayed on the same fourteen
shapes: **422 of 473 = 89.2%** on-geometry, with 3x fewer materialised samples
(and therefore 3x fewer cached latents).

WHY IT IS DEFAULT-OFF
=====================
Every one of the 24 published ai-toolkit ``config.yaml`` files in the Aug-3
tournament — including all 8 wins by the dominant operator 5FBmn1ax — used
``resolution: [512, 768, 1024]`` and set no ``bucket_tolerance``:

    krea2      11 artifacts  resolution [512, 768, 1024]  bucket_tolerance unset
    qwen_image  6 artifacts  resolution [512, 768, 1024]  bucket_tolerance unset
    zimage      4 artifacts  resolution [512, 768, 1024]  bucket_tolerance unset
    ideogram4   3 artifacts  resolution [512, 768, 1024]  bucket_tolerance unset

Nobody in the field deviates.  The claim that alignment helps is INFERRED from
the evaluator's source, not measured against a score, and it changes training
for every type at once.  So the switch is OFF unless explicitly opted in, per
type, exactly like `forge.tasks.holdout`:

    FORGE_EVAL_GEOMETRY_TYPES=krea2,z-image     # or "*" for all eligible

RESIDUAL MISMATCH THAT CANNOT BE REMOVED FROM CONFIG
====================================================
1. RESAMPLING FILTER.  ai-toolkit resizes with BICUBIC
   (dataloader_mixins.py:817); the evaluator uses LANCZOS.  Not configurable.
2. DIVISIBILITY.  ``bucket_tolerance`` in our config is DEAD — ``AiToolkitDataset
   .__init__`` overwrites it unconditionally with ``sd.get_bucket_divisibility()``
   (data_loader.py:395).  That is 16 for krea2/ideogram4/z-image (matching the
   evaluator's floor-to-16) but **32 for qwen-image and flux**.  Any evaluator
   size whose short edge is not a multiple of 32 is therefore UNREACHABLE for
   those two archs.  This is exactly why both Aug-3 flux tasks stay at 0%: their
   1195x896 sources score at 1024x752, and 752 is not a multiple of 32.  We
   still WRITE ``bucket_tolerance`` — set to the value the runtime will force —
   so the config states the truth rather than inheriting a lie.
3. HETEROGENEOUS ASPECT RATIOS.  One scalar resolution can be exact for one
   aspect-ratio family only.  A resolution LIST with one entry per family was
   considered and rejected: K entries means K dataset copies, so the
   on-geometry SHARE falls to 1/K while the sample count multiplies by K.  On
   the worst real shape (R3 krea2 db9f7244, three aspect ratios) a single
   resolution gives 26/43 = 60% on-geometry at 43 samples; the two-entry list
   gives 43/86 = 50% at 86 samples.  Single scalar wins on both axes.
4. SOURCES SMALLER THAN 1024 ON THE LONG EDGE.  The evaluator UPSCALES those to
   1024; ai-toolkit's bucket never upscales (``target = min(total, max)``), so
   they are unreachable at any resolution.  No Aug-3 source is in this class
   (all long edges are >= 1024), and the chooser degrades gracefully: such
   images simply never count as matches.

STRUCTURAL EXCLUSION: ideogram4
===============================
ideogram4 is INELIGIBLE and stays on the template list even when the switch
names it.  `forge.ideogram_release_policy` hash-binds the recipe projection, and
that projection INCLUDES ``dataset.resolution == [512, 768, 1024]``
(ideogram_release_policy.py:_EXPECTED_RECIPE).  Changing the resolution would:
  (a) make ``apply()`` a no-op, silently reverting lr 2.5e-5 -> 1e-4, dropping
      EMA/cosine/do_cfg/cache_latents — i.e. deleting the whole Week-5 recipe;
  (b) if applied after ``apply()``, make ``checkpoint_control()`` RAISE on the
      drifted projection, which in ``forge/tasks/aitoolkit.py`` propagates out of
      the handler and forfeits the task to the untrained fallback.
The activation record is bound to ``POLICY_SHA256``, so ``_EXPECTED_RECIPE``
cannot be edited without invalidating it.  Aligning ideogram4's geometry is a
separate, explicit decision that must go through the release-policy machinery.

Pure stdlib + optional Pillow.  Never raises (INV-1): every entry point degrades
to "no policy", which leaves the template list untouched.
"""

from __future__ import annotations

import math
import os
from collections import Counter

# --- the evaluator's contract ---------------------------------------------
EVAL_LONG_EDGE = 1024
EVAL_DIVISOR = 16

# ai-toolkit forces this per architecture at dataset construction
# (data_loader.py:395 -> sd.get_bucket_divisibility()).  Sources:
#   krea2      vae_scale_factor 8 * patch 2            (krea2.py:144-145,159-161)
#   ideogram4  vae_scale_factor 8 * patch 2            (ideogram4.py:181-182,207-209)
#   z-image    8 * 2                                   (z_image.py:64-65)
#   qwen-image 16 * 2                                  (qwen_image.py:81-82)
#   flux       2**(4-1) * 2 (is_flux) * 2 = 32, via the LEGACY StableDiffusion
#              class, because no registered BaseModel has arch == "flux"
#              (stable_diffusion_model.py:283-291; get_model.py:50-51)
BUCKET_DIVISIBILITY = {
    "krea2": 16,
    "ideogram4": 16,
    "z-image": 16,
    "qwen-image": 32,
    "flux": 32,
}

# ideogram4 is excluded structurally — see the module docstring.
_ELIGIBLE_TYPES = frozenset({"krea2", "z-image", "qwen-image", "flux"})

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Search window for the scalar resolution.  A bucket's long edge is
# R*sqrt(long/short) >= R, so R > EVAL_LONG_EDGE can never produce a 1024 long
# edge; and R below 64 is nonsense for these archs.  Exhaustive integer search
# over ~961 candidates against the DISTINCT source dimensions (typically 1-3)
# is microseconds, and exhaustive beats clever here: it is deterministic and has
# no rounding blind spot.
_R_MIN = 64
_R_MAX = EVAL_LONG_EDGE


def enabled_for(model_type: str) -> bool:
    """Whether the evaluator-geometry resolution policy applies to this type.

    Default OFF for every type: the entire Aug-3 field, winners included, shipped
    the inherited three-copy list, so this is an unvalidated deviation.  Opt in
    with ``FORGE_EVAL_GEOMETRY_TYPES`` (comma separated, or ``*``).
    """
    mt = (model_type or "").strip().lower()
    if mt not in _ELIGIBLE_TYPES:
        return False
    raw = os.environ.get("FORGE_EVAL_GEOMETRY_TYPES", "")
    allowed = {value.strip().lower() for value in raw.split(",") if value.strip()}
    return "*" in allowed or mt in allowed


def evaluator_size(width: int, height: int) -> tuple[int, int]:
    """Exact port of the evaluator's ``adjust_image_size``.

    ``evaluator/image_io.py:16-40``: long edge -> 1024 (LANCZOS), both dims
    floored to a multiple of 16, centre crop.  The crop is ``min(resized, floor)``
    which is always the floored value, so the scored size is deterministic in
    (w, h) alone.
    """
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("non-positive image dimension")
    if w > h:
        new_w = EVAL_LONG_EDGE
        new_h = int((h / w) * EVAL_LONG_EDGE)
    else:
        new_h = EVAL_LONG_EDGE
        new_w = int((w / h) * EVAL_LONG_EDGE)
    floor_w = (new_w // EVAL_DIVISOR) * EVAL_DIVISOR
    floor_h = (new_h // EVAL_DIVISOR) * EVAL_DIVISOR
    return (min(new_w, floor_w), min(new_h, floor_h))


def bucket_size(
    width: int, height: int, resolution: int, divisibility: int
) -> tuple[int, int]:
    """Exact port of ai-toolkit ``toolkit/buckets.py:get_bucket_for_image_size``.

    Caps TOTAL pixels at ``resolution**2`` (never upscales), preserves aspect,
    then picks whichever floor/ceil corner on the divisibility grid is closest in
    area to the target without exceeding the cap.  This is the true training
    geometry: ``BucketsMixin`` assigns ``crop_width/crop_height`` from it
    (dataloader_mixins.py:258-277).
    """
    w = int(width)
    h = int(height)
    total = w * h
    max_pixels = int(resolution) * int(resolution)
    target = min(total, max_pixels)
    scaler = (target / total) ** 0.5
    w_raw = (w * scaler) / divisibility
    h_raw = (h * scaler) / divisibility
    candidates = [
        (math.floor(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.floor(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
    ]
    capped = [
        (a, b) for a, b in candidates if a > 0 and b > 0 and a * b <= max_pixels
    ]
    if not capped:
        capped = [
            (
                max(divisibility, math.floor(w_raw) * divisibility),
                max(divisibility, math.floor(h_raw) * divisibility),
            )
        ]
    return min(capped, key=lambda wh: abs(wh[0] * wh[1] - target))


def choose_resolution(dims, divisibility: int) -> tuple[int, int, int]:
    """Return ``(resolution, matched_images, total_images)``.

    ``dims`` is any iterable of ``(width, height)`` — repeats are the weights, so
    the choice follows the dataset's actual composition.

    Ranking is fully deterministic:
      1. most images whose bucket EQUALS the evaluator's scored size;
      2. then the smallest mean ``|log(bucket_px / eval_px)|`` over ALL images,
         so the images that cannot match still land as close as possible in
         scale (this is what keeps flux useful at 0 exact matches: 992x768 is
         1.01x the scored pixel count where the incumbent ``@1024`` copy is
         1.34x);
      3. then the smallest resolution, which is the more robust tie-break — a
         smaller total-pixel cap cannot accidentally leave a larger source
         un-downscaled above the scored geometry.
    """
    counts = Counter((int(w), int(h)) for w, h in dims)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("no image dimensions")
    targets = {wh: evaluator_size(*wh) for wh in counts}

    best_key = None
    best_r = _R_MAX
    best_matched = 0
    for r in range(_R_MIN, _R_MAX + 1):
        matched = 0
        err = 0.0
        for wh, count in counts.items():
            ev = targets[wh]
            bk = bucket_size(wh[0], wh[1], r, divisibility)
            if bk == ev:
                matched += count
            err += count * abs(
                math.log((bk[0] * bk[1]) / (ev[0] * ev[1]))
            )
        key = (-matched, err / total, r)
        if best_key is None or key < best_key:
            best_key = key
            best_r = r
            best_matched = matched
    return best_r, best_matched, total


def measure_images(images_dir: str) -> list[tuple[int, int]]:
    """Read (width, height) for every image in the prepared dataset directory.

    Uses Pillow (already a hard dependency of this package, and present in the
    trainer image because ai-toolkit needs it).  Returns [] on any failure so
    callers degrade to "no policy" rather than forfeiting.
    """
    try:
        from PIL import Image
    except Exception:
        return []
    out: list[tuple[int, int]] = []
    try:
        names = sorted(
            e for e in os.listdir(images_dir) if e.lower().endswith(_IMAGE_EXTS)
        )
    except Exception:
        return []
    for name in names:
        try:
            with Image.open(os.path.join(images_dir, name)) as im:
                w, h = im.size
            if w > 0 and h > 0:
                out.append((int(w), int(h)))
        except Exception:
            continue
    return out


def plan(model_type: str, dims) -> dict | None:
    """Build the resolution policy record for a type + dataset shape.

    Returns ``None`` when the policy does not apply (ineligible type, no
    measurable images, or any internal error), in which case the caller MUST
    leave the template's ``resolution`` untouched.  Never raises (INV-1).
    """
    try:
        mt = (model_type or "").strip().lower()
        divisibility = BUCKET_DIVISIBILITY.get(mt)
        if divisibility is None:
            return None
        dims = [
            (int(w), int(h))
            for w, h in dims
            if int(w) > 0 and int(h) > 0
        ]
        if not dims:
            return None
        resolution, matched, total = choose_resolution(dims, divisibility)
        counts = Counter(dims)
        cohorts = []
        for wh in sorted(counts):
            ev = evaluator_size(*wh)
            bk = bucket_size(wh[0], wh[1], resolution, divisibility)
            cohorts.append(
                {
                    "source": f"{wh[0]}x{wh[1]}",
                    "count": counts[wh],
                    "evaluator": f"{ev[0]}x{ev[1]}",
                    "bucket": f"{bk[0]}x{bk[1]}",
                    "on_geometry": bk == ev,
                    # Why an unreachable cohort is unreachable, so the telemetry
                    # explains itself without re-deriving the divisibility rule.
                    "unreachable_reason": _unreachable_reason(
                        wh, ev, divisibility
                    ),
                }
            )
        return {
            "schema": 1,
            "model_type": mt,
            "resolution": int(resolution),
            "bucket_tolerance": int(divisibility),
            "images": total,
            "on_geometry": matched,
            "on_geometry_pct": round(100.0 * matched / total, 1),
            "cohorts": cohorts,
        }
    except Exception:
        return None


def _unreachable_reason(source, evaluator, divisibility: int) -> str | None:
    """Name the structural blocker for a cohort that cannot ever match, or None.

    ``None`` means "reachable in principle" — either it matched, or a different
    aspect-ratio cohort won the single-resolution tie-break.
    """
    try:
        w, h = source
        ev_w, ev_h = evaluator
        if ev_w % divisibility or ev_h % divisibility:
            return f"evaluator_size_not_multiple_of_{divisibility}"
        if w * h < ev_w * ev_h:
            return "source_smaller_than_scored_size"
        return None
    except Exception:
        return None
