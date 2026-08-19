"""Eval-grid base snap — train against (approximately) the base the EVALUATOR loads.

THE MISMATCH THIS ADDRESSES (week-9 base-parity probe, 2026-08-18)
==================================================================
Our trainer fine-tunes against the task repo the validator stages, but the
evaluator loads a DIFFERENT artifact (G.O.D validator/evaluation/evaluators/
diffusion.py:172-189 ``resolve_eval_base_model``, constants.py:18-21):

  * ideogram4 — trainer: bf16 PER-CHANNEL dequant of
    gradients-io-tournaments/ideogram-4-fp8 (ai-toolkit @ 99be3d96
    ``_dequantize_fp8_state_dict``); evaluator: Comfy-Org/Ideogram-4
    ``ideogram4_fp8_scaled.safetensors`` = per-tensor e4m3 at scale 1.0
    (all 211 scales OBSERVED == 1.0). Measured delta: 3.5-4.2% rel RMS,
    >99% of entries differ — 3-4x a typical rank-32 LoRA delta, in the lane
    whose DualModelGuider amplifies the adapter ~8x against an un-LoRA'd
    negative branch.
  * krea2 — trainer: krea/Krea-2-Raw ``raw.safetensors`` bf16 verbatim
    (the krea2 loader does NO dequant); evaluator: Comfy-Org/Krea-2
    ``krea2_raw_fp8_scaled.safetensors`` = per-tensor e4m3 with REAL scalar
    scales (NOT 1.0). Measured delta: ~2.7% rel RMS, 100% of entries differ.

Snapping the training base onto the eval e4m3 grid before ai-toolkit loads it
removes that fixed perturbation: measured reconstruction of the true eval
weights is 98.2-98.8% of entries for ideogram4 (probe align_check.py) and
99.999-100% for krea2 (this lane's spot checks — raw.safetensors is evidently
the exact master Comfy-Org quantized).

WHAT THIS MODULE DOES
=====================
``prepare(spec, deadline)`` builds a SHADOW copy of the staged base model in a
scratch directory with the transformer weights snapped onto the eval e4m3 grid,
and returns a path for ``build_config(..., base_model_override=...)`` to point
``model.name_or_path`` at. The staged /cache snapshot is NEVER mutated (it is
validator-owned and possibly shared). Everything else the loader reads
(index.json, vae/, configs) is symlinked from the original.

  * ideogram4: shadow dir; the fp8 shard is rewritten with
    codes' = nearest_e4m3(codes * per_channel_scale) and all scale vectors set
    to ones — so ai-toolkit's unchanged dequant path yields exactly the
    unit-scale e4m3 grid the evaluator loads. Header bytes are copied verbatim
    (dtypes/shapes/offsets unchanged).
  * krea2: shadow dir holding one rewritten ``raw.safetensors``; each of the
    256 eval-quantized bf16 weights becomes bf16(nearest_e4m3(W/s) * s) with s
    the eval file's baked per-tensor scale (forge/evalgrid_constants.py).

QUANTIZER SEMANTICS (matching the eval grid the probe measured)
===============================================================
float8_e4m3fn value set: sign, 4 exponent bits (bias 7), 3 mantissa bits;
normals ±2^-6..±448, subnormals ±(m/8)·2^-6 (min ±2^-9), NO infinities, NaN =
S.1111.111. Rounding: round-to-nearest, ties to even mantissa (RNE — matches
torch's float8_e4m3fn cast; a tie census on 13.1M real krea2 elements found 0
exact midpoints, so the tie policy is empirically irrelevant). Saturation:
|x| > 448 clamps to ±448 (absmax-scaled quantization semantics; also what the
comfy artifacts imply — max|W/s| observed exactly 448.0). Non-finite input
anywhere in a tensor ABORTS the snap (fail-open): a base with NaN/Inf is not a
base we should silently rewrite.

DEFAULT OFF / FAIL-OPEN / TIME CAP
==================================
* DEFAULT OFF for every type. Opt in per type with
  ``FORGE_EVALGRID_SNAP_TYPES=ideogram4,krea2`` (or ``*`` = all supported).
  Only ideogram4 and krea2 are supported; other types are never eligible.
  Flag off ⇒ byte-identical behavior: ``prepare`` returns None before touching
  the filesystem or telemetry.
* FAIL-OPEN: any error, unexpected layout, non-finite weight, insufficient
  disk, or cap breach logs loudly, deletes the partial shadow, and returns
  None — training proceeds on the original base exactly as today. This module
  must never kill a task.
* TIME CAP: ``FORGE_EVALGRID_SNAP_CAP_S`` (default 600 s), additionally bounded
  by 25% of the remaining wall clock when a deadline is provided; breach ⇒
  fail-open. The snap runs BEFORE the recipe's hours/step budget is computed
  (forge/tasks/aitoolkit.py), so time it consumes automatically shrinks the
  planned step count instead of being eaten by the end-of-run kill.
* DISK: the shadow is a copy (9.3 GB ideogram4 / 26.3 GB krea2). In-place
  rewrite of the staged snapshot would need zero headroom but mutates
  validator-owned, possibly task-shared state and cannot be made crash-safe —
  rejected. We require free_bytes >= source size + 1 GiB margin, else fail-open.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import time

from forge import telemetry

_TYPES_ENV = "FORGE_EVALGRID_SNAP_TYPES"
_CAP_ENV = "FORGE_EVALGRID_SNAP_CAP_S"
_DIR_ENV = "FORGE_EVALGRID_SNAP_DIR"

_DEFAULT_CAP_S = 600.0
# Never let the snap consume more than this fraction of the remaining clock,
# and skip outright when the effective cap would be smaller than the floor
# (a snap that must finish in <60s on a cold page cache is a lost bet).
_DEADLINE_FRACTION = 0.25
_MIN_CAP_S = 60.0
_DEFAULT_SCRATCH = "/tmp/forge-evalgrid"
_DISK_MARGIN_BYTES = 1 << 30  # 1 GiB
_COPY_CHUNK = 64 << 20  # 64 MiB streaming copy for untouched tensors

_SUPPORTED = ("ideogram4", "krea2")

_E4M3_MAX = 448.0

# Filenames fixed by the loaders at ai-toolkit pin 99be3d96 (OBSERVED):
_IDEOGRAM4_SHARD = "diffusion_pytorch_model.safetensors"
_KREA2_FILENAME = "raw.safetensors"


class _SnapAbort(Exception):
    """Internal: any condition that must fail open (reason in args[0])."""


# ---------------------------------------------------------------------------
# e4m3fn quantizer (pure numpy; the semantic core, unit-tested against a hand
# table and against torch.float8_e4m3fn where torch is available)
# ---------------------------------------------------------------------------

_GRID = None  # lazy: (values_f64[254?], codes_u8, mids_f64, tie_bump_bool)


def _e4m3_value_table(np):
    """All 256 e4m3fn code values as float64; NaN for S.1111.111."""
    out = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            out[b] = np.nan
        elif e == 0:
            out[b] = s * (m / 8.0) * 2.0 ** (-6)
        else:
            out[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return out


def _grid(np):
    """Sorted distinct finite e4m3 values, a canonical code per value, the
    midpoints between neighbours, and the RNE tie direction at each midpoint."""
    global _GRID
    if _GRID is not None:
        return _GRID
    tbl = _e4m3_value_table(np)
    finite = ~np.isnan(tbl)
    codes = np.arange(256, dtype=np.uint8)[finite]
    vals = tbl[finite]
    order = np.argsort(vals, kind="stable")
    vals, codes = vals[order], codes[order]
    # collapse the duplicate zero (+0.0 at 0x00, -0.0 at 0x80) to +0.0's code
    keep = np.ones(vals.size, dtype=bool)
    keep[1:] = vals[1:] != vals[:-1]
    dup = np.flatnonzero(~keep)
    for i in dup:  # canonical = the code with the sign bit clear (0x00)
        pick = min(codes[i - 1], codes[i])
        codes[i - 1] = pick
    vals, codes = vals[keep], codes[keep]
    mids = (vals[:-1] + vals[1:]) / 2.0
    # ties to EVEN mantissa: bump up exactly when the lower neighbour's
    # mantissa LSB is odd (sign bit does not participate in parity)
    tie_bump = (codes[:-1] & 1).astype(bool)
    _GRID = (vals, codes, mids, tie_bump)
    return _GRID


def nearest_e4m3_codes(x, np):
    """Nearest-e4m3fn codes of a float array: clamp to ±448 (saturation), then
    round to nearest with ties to even mantissa. Raises ValueError on any
    non-finite input — callers treat that as a fail-open condition."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite values are not snappable")
    vals, codes, mids, tie_bump = _grid(np)
    c = np.clip(x, -_E4M3_MAX, _E4M3_MAX)
    idx = np.searchsorted(mids, c)  # side='left': an exact midpoint -> lower
    at_mid = idx < mids.size
    at_mid &= c == mids[np.minimum(idx, mids.size - 1)]
    idx = idx + (at_mid & tie_bump[np.minimum(idx, mids.size - 1)])
    return codes[idx]


def e4m3_decode(codes_u8, np):
    """Code bytes -> float64 values (NaN for the NaN codes)."""
    return _e4m3_value_table(np)[np.asarray(codes_u8, dtype=np.uint8)]


def _f32_to_bf16_u16(x_f32, np):
    """IEEE round-to-nearest-even f32 -> bf16 bit patterns (uint16)."""
    u = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32)
    r = u + 0x7FFF + ((u >> 16) & 1)
    return (r >> 16).astype(np.uint16)


def _bf16_u16_to_f32(u16, np):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


# ---------------------------------------------------------------------------
# flag / cap / scratch plumbing
# ---------------------------------------------------------------------------


def enabled_for(model_type) -> bool:
    """Default OFF for every type; opt in per type (or '*') — only the two
    supported types are ever eligible."""
    mt = (model_type or "").strip().lower()
    if mt not in _SUPPORTED:
        return False
    raw = os.environ.get(_TYPES_ENV, "")
    allowed = {v.strip().lower() for v in raw.split(",") if v.strip()}
    return "*" in allowed or mt in allowed


def _cap_seconds(deadline) -> float:
    try:
        cap = float(os.environ.get(_CAP_ENV, "") or _DEFAULT_CAP_S)
    except (TypeError, ValueError):
        cap = _DEFAULT_CAP_S
    if deadline is not None:
        try:
            cap = min(cap, _DEADLINE_FRACTION * float(deadline.remaining()))
        except Exception:
            pass
    return cap


def _check_cap(started: float, cap: float) -> None:
    if time.monotonic() - started > cap:
        raise _SnapAbort("cap_exceeded")


def _scratch_root() -> str:
    return os.environ.get(_DIR_ENV, "") or _DEFAULT_SCRATCH


def _free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


# ---------------------------------------------------------------------------
# minimal safetensors plumbing (stdlib only; we keep header bytes VERBATIM so
# every untouched byte range stays byte-identical to the source)
# ---------------------------------------------------------------------------


def _read_header(path: str):
    """Return (header_dict_without_metadata, header_bytes_incl_prefix, data_start)."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise _SnapAbort("unexpected_layout")
        (n,) = struct.unpack("<Q", prefix)
        if n <= 0 or 8 + n > size:
            raise _SnapAbort("unexpected_layout")
        raw = fh.read(n)
        if len(raw) != n:
            raise _SnapAbort("unexpected_layout")
    try:
        hdr = json.loads(raw)
    except ValueError:
        raise _SnapAbort("unexpected_layout")
    if not isinstance(hdr, dict):
        raise _SnapAbort("unexpected_layout")
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    data_start = 8 + n
    # sanity: offsets contiguous-orderable and inside the file
    for k, v in tensors.items():
        try:
            a, b = v["data_offsets"]
        except (KeyError, TypeError, ValueError):
            raise _SnapAbort("unexpected_layout")
        if not (0 <= a <= b) or data_start + b > size:
            raise _SnapAbort("unexpected_layout")
    return tensors, prefix + raw, data_start


def _rewrite_safetensors(src, dst, transforms, np, started, cap) -> int:
    """Stream src -> dst copying header + every byte range verbatim, except
    tensors named in ``transforms`` whose bytes are replaced (same length).
    Returns the number of transformed tensors."""
    tensors, header_bytes, data_start = _read_header(src)
    missing = [k for k in transforms if k not in tensors]
    if missing:
        raise _SnapAbort("unexpected_layout")
    order = sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])
    src_size = os.path.getsize(src)
    done = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(header_bytes)
        pos = data_start
        for name, info in order:
            _check_cap(started, cap)
            a = data_start + info["data_offsets"][0]
            b = data_start + info["data_offsets"][1]
            if a < pos:
                raise _SnapAbort("unexpected_layout")  # overlapping tensors
            _copy_range(fin, fout, pos, a, started, cap)
            fin.seek(a)
            if name in transforms:
                raw = fin.read(b - a)
                if len(raw) != b - a:
                    raise _SnapAbort("unexpected_layout")
                out = transforms[name](raw, info, np)
                if len(out) != len(raw):
                    raise _SnapAbort("unexpected_layout")
                fout.write(out)
                done += 1
            else:
                _copy_range(fin, fout, a, b, started, cap)
            pos = b
        _copy_range(fin, fout, pos, src_size, started, cap)
    if os.path.getsize(dst) != src_size:
        raise _SnapAbort("unexpected_layout")
    return done


def _copy_range(fin, fout, start, end, started, cap) -> None:
    fin.seek(start)
    left = end - start
    while left > 0:
        _check_cap(started, cap)
        chunk = fin.read(min(_COPY_CHUNK, left))
        if not chunk:
            raise _SnapAbort("unexpected_layout")
        fout.write(chunk)
        left -= len(chunk)


# ---------------------------------------------------------------------------
# per-type transforms
# ---------------------------------------------------------------------------


def _ideogram4_weight_transform(scale_raw: bytes):
    """codes' = nearest_e4m3(codes * per_channel_scale); implemented as a
    256-entry code map per output channel (exact — the input per row can only
    take the 256 values TBL[c]*scale[r]) followed by one uint8 gather."""

    def transform(raw, info, np):
        rows = int(info["shape"][0])
        cols = int(info["shape"][1])
        codes = np.frombuffer(raw, dtype=np.uint8)
        if codes.size != rows * cols:
            raise _SnapAbort("unexpected_layout")
        if np.any((codes & 0x7F) == 0x7F):  # e4m3 NaN codes
            raise _SnapAbort("nonfinite_weights")
        scale = np.frombuffer(scale_raw, dtype=np.float32)
        if scale.size != rows or not np.all(np.isfinite(scale)):
            raise _SnapAbort("unexpected_layout")
        if float(np.max(np.abs(scale))) > 1.0:
            # dequant would exceed ±448 and saturate wholesale — not the
            # artifact family this transform was measured on
            raise _SnapAbort("unexpected_layout")
        tbl = _e4m3_value_table(np)
        products = tbl[None, :] * scale.astype(np.float64)[:, None]  # (rows,256)
        products[:, [0x7F, 0xFF]] = 0.0  # NaN slots never indexed (checked above)
        rowmap = nearest_e4m3_codes(products, np).reshape(rows, 256)
        idx = codes.astype(np.int32)
        idx += (np.arange(rows, dtype=np.int32) * 256).repeat(cols)
        return rowmap.reshape(-1)[idx].astype(np.uint8).tobytes()

    return transform


def _ideogram4_scale_transform(raw, info, np):
    """The shadow shard advertises the eval grid's universal scale: ones."""
    n = 1
    for d in info["shape"]:
        n *= int(d)
    return np.ones(n, dtype=np.float32).tobytes()


def _krea2_transform(eval_scale):
    """bf16 W -> bf16(nearest_e4m3(W/s) * s); implemented as a 65536-entry
    bf16->bf16 lookup (exact) followed by one uint16 gather."""

    def transform(raw, info, np):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        if np.any((u16 & 0x7F80) == 0x7F80):  # bf16 Inf/NaN patterns
            raise _SnapAbort("nonfinite_weights")
        all_f32 = _bf16_u16_to_f32(np.arange(65536, dtype=np.uint16), np).copy()
        all_f32[~np.isfinite(all_f32)] = 0.0  # never indexed (checked above);
        # zero BEFORE any float op so signaling-NaN bit patterns are never cast
        x = all_f32.astype(np.float64) / float(eval_scale)
        snapped = e4m3_decode(nearest_e4m3_codes(x, np), np) * float(eval_scale)
        table = _f32_to_bf16_u16(snapped.astype(np.float32), np)
        return table[u16].tobytes()

    return transform


# ---------------------------------------------------------------------------
# shadow builders
# ---------------------------------------------------------------------------


def _ensure_scratch(base_dir: str, mt: str, task_id: str, src_bytes: int) -> str:
    task_key = hashlib.sha256((task_id or "").encode("utf-8")).hexdigest()[:12]
    root = _scratch_root()
    shadow = os.path.join(root, f"{mt}-{task_key}")
    if os.path.lexists(shadow):
        shutil.rmtree(shadow, ignore_errors=True)
    os.makedirs(shadow, exist_ok=True)
    if _free_bytes(shadow) < src_bytes + _DISK_MARGIN_BYTES:
        raise _SnapAbort("insufficient_disk")
    return shadow


def _symlink_siblings(src_dir: str, dst_dir: str, skip: str) -> None:
    for entry in os.listdir(src_dir):
        if entry == skip:
            continue
        os.symlink(os.path.join(src_dir, entry), os.path.join(dst_dir, entry))


def _build_ideogram4_shadow(spec, np, started, cap) -> tuple[str, int]:
    from forge.evalgrid_constants import IDEOGRAM4_FP8_TENSORS

    base = spec.cached_model_dir
    tdir = os.path.join(base, "transformer")
    shard = os.path.join(tdir, _IDEOGRAM4_SHARD)
    if not os.path.isfile(shard):
        raise _SnapAbort("unexpected_layout")
    tensors, _, data_start = _read_header(shard)

    fp8 = {k for k, v in tensors.items() if v.get("dtype") == "F8_E4M3"}
    if fp8 != set(IDEOGRAM4_FP8_TENSORS):
        raise _SnapAbort("unexpected_layout")
    scale_names = {}
    for name in fp8:
        if tuple(tensors[name]["shape"]) != IDEOGRAM4_FP8_TENSORS[name]:
            raise _SnapAbort("unexpected_layout")
        sk = name + "_scale"
        sv = tensors.get(sk)
        if (
            sv is None
            or sv.get("dtype") != "F32"
            or tuple(sv["shape"]) != (IDEOGRAM4_FP8_TENSORS[name][0],)
        ):
            raise _SnapAbort("unexpected_layout")
        scale_names[name] = sk

    # read the (small) scale vectors up front so each weight transform can fold
    # its own scale without a second pass over the file
    transforms = {}
    with open(shard, "rb") as fh:
        for name, sk in scale_names.items():
            a, b = tensors[sk]["data_offsets"]
            fh.seek(data_start + a)
            transforms[name] = _ideogram4_weight_transform(fh.read(b - a))
            transforms[sk] = _ideogram4_scale_transform

    shadow = _ensure_scratch(
        base, "ideogram4", spec.task_id, os.path.getsize(shard)
    )
    try:
        shadow_tdir = os.path.join(shadow, "transformer")
        os.makedirs(shadow_tdir, exist_ok=True)
        done = _rewrite_safetensors(
            shard, os.path.join(shadow_tdir, _IDEOGRAM4_SHARD), transforms, np,
            started, cap,
        )
        _symlink_siblings(tdir, shadow_tdir, skip=_IDEOGRAM4_SHARD)
        _symlink_siblings(base, shadow, skip="transformer")
    except BaseException:
        shutil.rmtree(shadow, ignore_errors=True)
        raise
    return shadow, done


def _build_krea2_shadow(spec, np, started, cap) -> tuple[str, int]:
    from forge.evalgrid_constants import KREA2_FP8_TENSORS

    base = spec.cached_model_dir
    if not os.path.isdir(base):
        raise _SnapAbort("unexpected_layout")
    # mirror the loader's selection exactly: the lone *.safetensors in the dir
    candidates = [f for f in os.listdir(base) if f.endswith(".safetensors")]
    if len(candidates) != 1:
        raise _SnapAbort("unexpected_layout")
    src = os.path.join(base, candidates[0])
    tensors, _, _ = _read_header(src)

    transforms = {}
    for name, (scale, shape) in KREA2_FP8_TENSORS.items():
        info = tensors.get(name)
        if (
            info is None
            or info.get("dtype") != "BF16"
            or tuple(info["shape"]) != tuple(shape)
        ):
            raise _SnapAbort("unexpected_layout")
        transforms[name] = _krea2_transform(scale)

    shadow = _ensure_scratch(base, "krea2", spec.task_id, os.path.getsize(src))
    try:
        done = _rewrite_safetensors(
            src, os.path.join(shadow, candidates[0]), transforms, np, started,
            cap,
        )
    except BaseException:
        shutil.rmtree(shadow, ignore_errors=True)
        raise
    # name_or_path stays a DIRECTORY (the loader picks the lone *.safetensors),
    # so nothing else is needed in the shadow: vae_path/text_encoder_path are
    # injected separately by forge/config.py and keep pointing at the originals.
    return shadow, done


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def prepare(spec, deadline=None):
    """Build the snapped shadow base for ``spec`` if the per-type flag is on.

    Returns the path for ``model.name_or_path`` (a directory for both types) or
    None. NEVER raises; every failure logs loudly and falls open to the
    original staged base.
    """
    mt = (getattr(spec, "model_type", "") or "").strip().lower()
    if not enabled_for(mt):
        return None

    started = time.monotonic()
    shadow = None
    try:
        import numpy as np

        cap = _cap_seconds(deadline)
        if cap < _MIN_CAP_S:
            raise _SnapAbort("insufficient_budget")
        if mt == "ideogram4":
            shadow, done = _build_ideogram4_shadow(spec, np, started, cap)
        else:
            shadow, done = _build_krea2_shadow(spec, np, started, cap)
        elapsed = round(time.monotonic() - started, 1)
        telemetry.event(
            "evalgrid_snap_applied",
            model_type=mt,
            shadow_path=shadow,
            tensors_snapped=done,
            elapsed_s=elapsed,
            cap_s=round(cap, 1),
        )
        _log(f"{mt}: snapped {done} tensors onto the eval e4m3 grid in "
             f"{elapsed}s -> {shadow}")
        return shadow
    except BaseException as exc:  # noqa: BLE001 — fail-open is the contract
        reason = (
            exc.args[0]
            if isinstance(exc, _SnapAbort) and exc.args
            else "error"
        )
        if shadow:
            shutil.rmtree(shadow, ignore_errors=True)
        try:
            telemetry.event(
                "evalgrid_snap_inactive",
                model_type=mt,
                reason=reason,
                error_type=type(exc).__name__,
                elapsed_s=round(time.monotonic() - started, 1),
            )
        except Exception:
            pass
        _log(
            f"{mt}: snap FAILED OPEN ({reason}: {type(exc).__name__}: {exc}) — "
            "training on the ORIGINAL staged base"
        )
        return None


def _log(msg: str) -> None:
    print(f"[forge.evalgrid_snap] {msg}", file=sys.stderr, flush=True)
