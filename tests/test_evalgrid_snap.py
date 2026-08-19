"""Week-9 eval-grid base snap (forge/evalgrid_snap.py) — CPU tests.

Covers, per the lane's task contract:
  1. the e4m3fn quantizer against a HAND-COMPUTED table (values, saturation,
     subnormals, RNE ties, NaN refusal) and against an independent brute-force
     reference;
  2. torch cross-check (pytest.importorskip — the repo venv has no torch; the
     scratchpad validation venv runs it, version recorded in the lane
     CHANGES.md);
  3. flag OFF by default for every type, and flag-off byte-identity (no
     filesystem effect, no telemetry, identical emitted config);
  4. end-to-end snap on tiny staged-model fixtures for BOTH types, checking
     the exact bytes (header verbatim, untouched tensors verbatim, snapped
     codes/values equal to an independent reference, ideogram4 scales -> ones);
  5. fail-open on corrupt shard, unexpected layout, non-finite weights,
     insufficient disk, insufficient budget, and cap breach — original base
     untouched, partial shadow cleaned, loud telemetry;
  6. the build_config(base_model_override=...) plumbing (normal path only,
     krea2 vae_path stays on the original staged dir);
  7. provenance pins on the generated forge/evalgrid_constants.py.
"""

from __future__ import annotations

import json
import os
import struct
import time
import types

import numpy as np
import pytest
import yaml

from forge import evalgrid_constants, evalgrid_snap
from forge.config import build_config
from forge.data.schema import ImageSpec


# --------------------------------------------------------------------------- #
# independent reference quantizer (deliberately a different algorithm:
# brute-force nearest over the explicit finite code list, ties by mantissa
# parity)
# --------------------------------------------------------------------------- #

def _ref_code_values():
    """(code, value) for every finite e4m3fn code, from first principles."""
    out = []
    for b in range(256):
        s = -1.0 if b >> 7 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            continue  # NaN
        if e == 0:
            out.append((b, s * (m / 8.0) * 2.0 ** -6))
        else:
            out.append((b, s * (1 + m / 8.0) * 2.0 ** (e - 7)))
    return out


def _ref_nearest(x: float) -> float:
    """Round-to-nearest e4m3fn value, ties to even mantissa, saturating."""
    x = max(-448.0, min(448.0, x))
    best = None
    for code, v in _ref_code_values():
        d = abs(x - v)
        if best is None or d < best[0]:
            best = (d, v, code)
        elif d == best[0] and (code & 1) == 0 and (best[2] & 1) == 1:
            # exact tie: prefer the even mantissa (code LSB clear)
            best = (d, v, code)
    return best[1]


def _module_nearest_vals(xs):
    codes = evalgrid_snap.nearest_e4m3_codes(np.asarray(xs, dtype=np.float64), np)
    return evalgrid_snap.e4m3_decode(codes, np)


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return r.view(np.float32)


# --------------------------------------------------------------------------- #
# 1. quantizer semantics
# --------------------------------------------------------------------------- #

def test_quantizer_hand_computed_table():
    """Literal hand-computed e4m3fn facts — no code-derived expectations."""
    cases = [
        # exact grid values map to themselves
        (448.0, 448.0),          # max normal (1+6/8)*2^8
        (-448.0, -448.0),
        (0.0, 0.0),
        (0.001953125, 0.001953125),    # min positive subnormal 2^-9
        (-0.001953125, -0.001953125),
        (240.0, 240.0),          # (1+7/8)*2^7
        (1.75, 1.75),            # (1+6/8)*2^0
        # saturation
        (1000.0, 448.0),
        (-1e9, -448.0),
        (448.0001, 448.0),
        # nearest (not at a tie)
        (250.0, 256.0),          # grid 240, 256; midpoint 248 < 250
        (247.0, 240.0),
        (0.0015, 0.001953125),   # midpoint 0.0009765625 < 0.0015
        (0.0009, 0.0),           # below the 0-vs-2^-9 midpoint
        # RNE ties (exact midpoints; even-mantissa side wins)
        (432.0, 448.0),          # 416 (m=5, odd) vs 448 (m=6, even) -> up
        (400.0, 384.0),          # 384 (m=4, even) vs 416 (m=5, odd) -> down
        (-400.0, -384.0),        # sign does not affect mantissa parity
        (0.0009765625, 0.0),     # 2^-10: 0 (m=0, even) vs 2^-9 (m=1, odd)
        (-0.0009765625, 0.0),
    ]
    got = _module_nearest_vals([c[0] for c in cases])
    for (x, want), g in zip(cases, got):
        assert g == want, f"nearest({x}) = {g}, want {want}"


def test_quantizer_matches_bruteforce_reference_everywhere():
    grid = np.array([v for _, v in _ref_code_values()], dtype=np.float64)
    uvals = np.unique(grid)
    mids = (uvals[:-1] + uvals[1:]) / 2.0
    rng = np.random.default_rng(9)
    rand = rng.uniform(-500, 500, size=4096)
    small = rng.uniform(-0.02, 0.02, size=4096)
    xs = np.concatenate([uvals, mids, rand, small])
    got = _module_nearest_vals(xs)
    for x, g in zip(xs.tolist(), got.tolist()):
        assert g == _ref_nearest(x), f"nearest({x!r}) = {g!r}"


def test_quantizer_grid_is_idempotent_and_453_values():
    uvals = np.unique([v for _, v in _ref_code_values()])
    # 254 finite codes minus the duplicated zero = 253 distinct values
    assert uvals.size == 253
    got = _module_nearest_vals(uvals)
    assert np.array_equal(got, uvals)


def test_quantizer_refuses_nonfinite():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            evalgrid_snap.nearest_e4m3_codes(np.array([1.0, bad]), np)


def test_quantizer_matches_torch_float8_cast():
    """Cross-check against torch's float8_e4m3fn cast (RNE) where torch
    exists. The repo venv has no torch -> skipped there; the scratchpad
    validation venv runs it (torch version recorded in the lane CHANGES.md)."""
    torch = pytest.importorskip("torch")
    uvals = np.unique([v for _, v in _ref_code_values()])
    mids = (uvals[:-1] + uvals[1:]) / 2.0
    rng = np.random.default_rng(11)
    xs = np.concatenate(
        [uvals, mids, rng.uniform(-460, 460, 8192), rng.uniform(-0.02, 0.02, 8192)]
    ).astype(np.float32)  # exact f32 inputs so both sides see identical values
    ours = _module_nearest_vals(xs.astype(np.float64))
    theirs = (
        torch.from_numpy(xs)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .numpy()
        .astype(np.float64)
    )
    assert np.array_equal(ours, theirs)


# --------------------------------------------------------------------------- #
# fixture plumbing
# --------------------------------------------------------------------------- #

def _write_safetensors(path, tensors):
    """tensors: list of (name, dtype_str, shape, raw_bytes)."""
    header = {}
    off = 0
    for name, dt, shape, raw in tensors:
        header[name] = {
            "dtype": dt,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        off += len(raw)
    hb = json.dumps(header).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for _, _, _, raw in tensors:
            fh.write(raw)


def _read_safetensors(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(n))
        data = fh.read()
    return hdr, data, 8 + n


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


class _Spec(types.SimpleNamespace):
    pass


def _snap_env(monkeypatch, tmp_path, types_value):
    if types_value is None:
        monkeypatch.delenv("FORGE_EVALGRID_SNAP_TYPES", raising=False)
    else:
        monkeypatch.setenv("FORGE_EVALGRID_SNAP_TYPES", types_value)
    monkeypatch.setenv("FORGE_EVALGRID_SNAP_DIR", str(tmp_path / "scratch"))
    monkeypatch.delenv("FORGE_EVALGRID_SNAP_CAP_S", raising=False)


def _capture_telemetry(monkeypatch):
    events = []
    monkeypatch.setattr(
        evalgrid_snap.telemetry,
        "event",
        lambda name, **kv: events.append((name, kv)),
    )
    return events


_IDEO_TABLE = {"layers.0.w.weight": (4, 8), "layers.1.w.weight": (2, 4)}


def _ideogram4_fixture(tmp_path, monkeypatch, *, codes0=None, scales0=None):
    monkeypatch.setattr(
        evalgrid_constants, "IDEOGRAM4_FP8_TENSORS", dict(_IDEO_TABLE)
    )
    base = tmp_path / "staged-ideo"
    rng = np.random.default_rng(3)
    if codes0 is None:
        codes0 = rng.integers(0, 256, size=(4, 8), dtype=np.uint8)
        codes0[(codes0 & 0x7F) == 0x7F] = 0  # keep finite
    if scales0 is None:
        scales0 = np.array([1e-3, 2e-3, 5e-4, 1e-4], dtype=np.float32)
    codes1 = rng.integers(0, 128, size=(2, 4), dtype=np.uint8)
    codes1[(codes1 & 0x7F) == 0x7F] = 0
    scales1 = np.array([3e-4, 4e-4], dtype=np.float32)
    norm = rng.standard_normal(8).astype(np.float32)
    tensors = [
        ("layers.0.w.weight", "F8_E4M3", (4, 8), codes0.tobytes()),
        ("layers.0.w.weight_scale", "F32", (4,), scales0.tobytes()),
        ("layers.1.w.weight", "F8_E4M3", (2, 4), codes1.tobytes()),
        ("layers.1.w.weight_scale", "F32", (2,), scales1.tobytes()),
        ("norm.weight", "F32", (8,), norm.tobytes()),
    ]
    shard = base / "transformer" / "diffusion_pytorch_model.safetensors"
    _write_safetensors(str(shard), tensors)
    (base / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}})
    )
    (base / "transformer" / "config.json").write_text("{}")
    os.makedirs(base / "vae")
    (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae-bytes")
    (base / "model_index.json").write_text("{}")
    spec = _Spec(
        model_type="ideogram4", task_id="t-ideo", cached_model_dir=str(base)
    )
    return spec, {
        "codes": [codes0, codes1],
        "scales": [scales0, scales1],
        "shard": str(shard),
    }


_KREA_S0 = 0.00146484375
_KREA_S1 = 0.0078125
_KREA_TABLE = {"a.weight": (_KREA_S0, (4, 8)), "b.weight": (_KREA_S1, (2, 2))}


def _krea2_fixture(tmp_path, monkeypatch, *, w0=None):
    monkeypatch.setattr(evalgrid_constants, "KREA2_FP8_TENSORS", dict(_KREA_TABLE))
    base = tmp_path / "staged-krea"
    rng = np.random.default_rng(5)
    if w0 is None:
        w0 = _bf16_round((rng.standard_normal((4, 8)) * 0.3).astype(np.float32))
    w1 = _bf16_round((rng.standard_normal((2, 2)) * 2.0).astype(np.float32))
    norm = rng.standard_normal(4).astype(np.float32)

    def bf16_bytes(a):
        return (
            np.ascontiguousarray(a, dtype=np.float32)
            .view(np.uint32)
            .__rshift__(16)
            .astype(np.uint16)
            .tobytes()
        )

    tensors = [
        ("a.weight", "BF16", (4, 8), bf16_bytes(w0)),
        ("b.weight", "BF16", (2, 2), bf16_bytes(w1)),
        ("norm.scale", "F32", (4,), norm.tobytes()),
    ]
    _write_safetensors(str(base / "raw.safetensors"), tensors)
    spec = _Spec(model_type="krea2", task_id="t-krea", cached_model_dir=str(base))
    return spec, {"w": [w0, w1], "src": str(base / "raw.safetensors")}


# --------------------------------------------------------------------------- #
# 3. default-off + flag-off byte identity
# --------------------------------------------------------------------------- #

def test_flag_default_off_for_every_type(monkeypatch):
    monkeypatch.delenv("FORGE_EVALGRID_SNAP_TYPES", raising=False)
    for mt in ("flux", "krea2", "ideogram4", "z-image", "qwen-image"):
        assert evalgrid_snap.enabled_for(mt) is False


def test_flag_parsing_supported_types_only(monkeypatch):
    monkeypatch.setenv("FORGE_EVALGRID_SNAP_TYPES", "*")
    assert evalgrid_snap.enabled_for("ideogram4") is True
    assert evalgrid_snap.enabled_for("krea2") is True
    for mt in ("flux", "z-image", "qwen-image", "", None):
        assert evalgrid_snap.enabled_for(mt) is False
    monkeypatch.setenv("FORGE_EVALGRID_SNAP_TYPES", " Ideogram4 , nonsense ")
    assert evalgrid_snap.enabled_for("ideogram4") is True
    assert evalgrid_snap.enabled_for("krea2") is False


def test_flag_off_prepare_is_a_no_op(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, None)
    events = _capture_telemetry(monkeypatch)
    spec, _ = _ideogram4_fixture(tmp_path, monkeypatch)
    before = _read_bytes(_shard_of(spec))
    assert evalgrid_snap.prepare(spec) is None
    assert _read_bytes(_shard_of(spec)) == before
    assert not (tmp_path / "scratch").exists()  # nothing written anywhere
    assert events == []  # telemetry silent when the flag is off


def _shard_of(spec):
    return os.path.join(
        spec.cached_model_dir, "transformer", "diffusion_pytorch_model.safetensors"
    )


def test_flag_env_alone_does_not_change_the_emitted_config(monkeypatch, tmp_path):
    """The env var by itself (no override built) must be config-inert, and the
    override default (None) must leave name_or_path on the staged dir."""
    spec = ImageSpec.build(
        task_id="t1", model="krea/Krea-2-Raw", model_type="krea2",
        expected_repo_name="r", trigger_word=None, dataset_zip=None,
    )
    monkeypatch.delenv("FORGE_EVALGRID_SNAP_TYPES", raising=False)
    off = yaml.safe_dump(build_config(spec, 10, 1.0), sort_keys=False)
    monkeypatch.setenv("FORGE_EVALGRID_SNAP_TYPES", "*")
    on_no_override = yaml.safe_dump(build_config(spec, 10, 1.0), sort_keys=False)
    assert off == on_no_override
    cfg = yaml.safe_load(off)
    p = cfg["config"]["process"][0]
    assert p["model"]["name_or_path"] == spec.cached_model_dir


# --------------------------------------------------------------------------- #
# 4. end-to-end snap on fixtures
# --------------------------------------------------------------------------- #

def test_ideogram4_snap_end_to_end(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "ideogram4")
    events = _capture_telemetry(monkeypatch)
    spec, fx = _ideogram4_fixture(tmp_path, monkeypatch)
    src_bytes = _read_bytes(fx["shard"])

    shadow = evalgrid_snap.prepare(spec)
    assert shadow is not None
    assert [e[0] for e in events] == ["evalgrid_snap_applied"]
    assert events[0][1]["tensors_snapped"] == 4  # 2 weights + 2 scales

    # original staged shard untouched
    assert _read_bytes(fx["shard"]) == src_bytes

    out = os.path.join(shadow, "transformer", "diffusion_pytorch_model.safetensors")
    hdr, data, ds = _read_safetensors(out)
    src_hdr, src_data, src_ds = _read_safetensors(fx["shard"])
    assert hdr == src_hdr and ds == src_ds  # header bytes semantics preserved
    with open(out, "rb") as a, open(fx["shard"], "rb") as b:
        assert a.read(ds) == b.read(src_ds)  # header VERBATIM

    def tensor_bytes(hdr, data, name):
        a, b = hdr[name]["data_offsets"]
        return data[a:b]

    # snapped codes == independent reference: nearest e4m3 of code_val * scale
    for (name, _), codes, scales in zip(
        sorted(_IDEO_TABLE.items()), fx["codes"], fx["scales"]
    ):
        got = np.frombuffer(tensor_bytes(hdr, data, name), dtype=np.uint8).reshape(
            codes.shape
        )
        tbl = evalgrid_snap._e4m3_value_table(np)
        for r in range(codes.shape[0]):
            for c in range(codes.shape[1]):
                v = tbl[codes[r, c]] * float(scales[r])
                assert tbl[got[r, c]] == _ref_nearest(v)
        # the shadow advertises the eval grid's universal scale: ones
        s = np.frombuffer(
            tensor_bytes(hdr, data, name + "_scale"), dtype=np.float32
        )
        assert np.all(s == 1.0)

    # untouched tensor byte-identical
    assert tensor_bytes(hdr, data, "norm.weight") == tensor_bytes(
        src_hdr, src_data, "norm.weight"
    )
    # siblings symlinked, vae + top-level entries reachable
    tdir = os.path.join(shadow, "transformer")
    assert os.path.islink(
        os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")
    )
    assert os.path.islink(os.path.join(shadow, "vae"))
    assert (
        _read_bytes(os.path.join(shadow, "vae", "diffusion_pytorch_model.safetensors"))
        == b"vae-bytes"
    )


def test_krea2_snap_end_to_end(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "krea2")
    events = _capture_telemetry(monkeypatch)
    spec, fx = _krea2_fixture(tmp_path, monkeypatch)
    src_bytes = _read_bytes(fx["src"])

    shadow = evalgrid_snap.prepare(spec)
    assert shadow is not None
    assert [e[0] for e in events] == ["evalgrid_snap_applied"]
    assert events[0][1]["tensors_snapped"] == 2
    assert _read_bytes(fx["src"]) == src_bytes  # original untouched

    out = os.path.join(shadow, "raw.safetensors")
    hdr, data, _ = _read_safetensors(out)
    src_hdr, src_data, _ = _read_safetensors(fx["src"])

    def tensor(hdr, data, name, dt):
        a, b = hdr[name]["data_offsets"]
        raw = data[a:b]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32)
        return np.frombuffer(raw, dtype=np.float32)

    for (name, (s, shape)), w in zip(sorted(_KREA_TABLE.items()), fx["w"]):
        got = tensor(hdr, data, name, "BF16")
        want = _bf16_round(
            np.array(
                [_ref_nearest(float(v) / s) * s for v in w.reshape(-1)],
                dtype=np.float32,
            )
        )
        assert np.array_equal(got, want.reshape(-1)), name
    # non-quantized tensor byte-identical
    a, b = hdr["norm.scale"]["data_offsets"]
    sa, sb = src_hdr["norm.scale"]["data_offsets"]
    assert data[a:b] == src_data[sa:sb]

    # snapping is idempotent: run again on a spec pointing at the shadow
    spec2 = _Spec(model_type="krea2", task_id="t-krea-2", cached_model_dir=shadow)
    shadow2 = evalgrid_snap.prepare(spec2)
    assert shadow2 is not None
    assert _read_bytes(os.path.join(shadow2, "raw.safetensors")) == _read_bytes(out)


# --------------------------------------------------------------------------- #
# 5. fail-open
# --------------------------------------------------------------------------- #

def _assert_failed_open(events, spec, reason):
    assert [e[0] for e in events] == ["evalgrid_snap_inactive"]
    assert events[0][1]["reason"] == reason
    scratch = os.environ["FORGE_EVALGRID_SNAP_DIR"]
    if os.path.isdir(scratch):  # no partial shadow left behind
        for entry in os.listdir(scratch):
            assert not os.listdir(os.path.join(scratch, entry)), entry


def test_fail_open_corrupt_shard_truncated(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "ideogram4")
    events = _capture_telemetry(monkeypatch)
    spec, fx = _ideogram4_fixture(tmp_path, monkeypatch)
    raw = _read_bytes(fx["shard"])
    with open(fx["shard"], "wb") as fh:
        fh.write(raw[: len(raw) // 2])  # truncate the data
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "unexpected_layout")


def test_fail_open_corrupt_shard_garbage_header(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "krea2")
    events = _capture_telemetry(monkeypatch)
    spec, fx = _krea2_fixture(tmp_path, monkeypatch)
    with open(fx["src"], "wb") as fh:
        fh.write(struct.pack("<Q", 40) + b"this is not json" + b"\0" * 40)
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "unexpected_layout")


def test_fail_open_unexpected_tensor_set(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "ideogram4")
    events = _capture_telemetry(monkeypatch)
    spec, _ = _ideogram4_fixture(tmp_path, monkeypatch)
    # baked table no longer matches the shard's fp8 set
    monkeypatch.setattr(
        evalgrid_constants,
        "IDEOGRAM4_FP8_TENSORS",
        {**_IDEO_TABLE, "layers.9.w.weight": (4, 4)},
    )
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "unexpected_layout")


def test_fail_open_nonfinite_weights(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "ideogram4")
    events = _capture_telemetry(monkeypatch)
    codes = np.full((4, 8), 0x7F, dtype=np.uint8)  # e4m3 NaN pattern
    spec, _ = _ideogram4_fixture(tmp_path, monkeypatch, codes0=codes)
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "nonfinite_weights")


def test_fail_open_nonfinite_bf16(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "krea2")
    events = _capture_telemetry(monkeypatch)
    w = np.full((4, 8), np.inf, dtype=np.float32)
    spec, _ = _krea2_fixture(tmp_path, monkeypatch, w0=w)
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "nonfinite_weights")


def test_fail_open_insufficient_disk(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "krea2")
    events = _capture_telemetry(monkeypatch)
    spec, _ = _krea2_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(evalgrid_snap, "_free_bytes", lambda path: 0)
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "insufficient_disk")


def test_fail_open_cap_breach(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "krea2")
    events = _capture_telemetry(monkeypatch)
    spec, fx = _krea2_fixture(tmp_path, monkeypatch)
    src_bytes = _read_bytes(fx["src"])

    # a clock that jumps far past the cap on every look after the start
    ticks = iter([0.0] + [10_000.0] * 1000)
    fake = types.SimpleNamespace(monotonic=lambda: next(ticks))
    monkeypatch.setattr(evalgrid_snap, "time", fake)

    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "cap_exceeded")
    assert _read_bytes(fx["src"]) == src_bytes


def test_fail_open_insufficient_budget_cap_env(monkeypatch, tmp_path):
    _snap_env(monkeypatch, tmp_path, "krea2")
    monkeypatch.setenv("FORGE_EVALGRID_SNAP_CAP_S", "30")  # below the 60s floor
    events = _capture_telemetry(monkeypatch)
    spec, _ = _krea2_fixture(tmp_path, monkeypatch)
    assert evalgrid_snap.prepare(spec) is None
    _assert_failed_open(events, spec, "insufficient_budget")


def test_fail_open_insufficient_budget_deadline(monkeypatch, tmp_path):
    from forge.clock import Deadline

    _snap_env(monkeypatch, tmp_path, "krea2")
    events = _capture_telemetry(monkeypatch)
    spec, _ = _krea2_fixture(tmp_path, monkeypatch)
    # 100s remaining -> 25% = 25s effective cap < 60s floor -> skip
    deadline = Deadline.from_hours(
        100.0 / 3600.0, started_monotonic=time.monotonic(), export_reserve_s=0.0
    )
    assert evalgrid_snap.prepare(spec, deadline) is None
    _assert_failed_open(events, spec, "insufficient_budget")


def test_cap_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("FORGE_EVALGRID_SNAP_CAP_S", "not-a-number")
    assert evalgrid_snap._cap_seconds(None) == 600.0
    monkeypatch.delenv("FORGE_EVALGRID_SNAP_CAP_S", raising=False)
    assert evalgrid_snap._cap_seconds(None) == 600.0


# --------------------------------------------------------------------------- #
# 6. build_config plumbing
# --------------------------------------------------------------------------- #

def test_build_config_applies_override_and_keeps_krea2_vae_on_original():
    spec = ImageSpec.build(
        task_id="t1", model="krea/Krea-2-Raw", model_type="krea2",
        expected_repo_name="r", trigger_word=None, dataset_zip=None,
    )
    cfg = build_config(spec, 10, 1.0, base_model_override="/tmp/shadow-x")
    p = cfg["config"]["process"][0]
    assert p["model"]["name_or_path"] == "/tmp/shadow-x"
    # only the transformer is snapped: VAE + TE stay on the staged originals
    assert p["model"]["model_kwargs"]["vae_path"] == spec.cached_model_dir
    assert p["model"]["model_kwargs"]["text_encoder_path"] == (
        "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct"
    )


def test_build_config_override_survives_ideogram_release_policy():
    spec = ImageSpec.build(
        task_id="t1", model="gradients-io-tournaments/ideogram-4-fp8",
        model_type="ideogram4", expected_repo_name="r", trigger_word=None,
        dataset_zip=None,
    )
    cfg = build_config(spec, 16, 1.0, base_model_override="/tmp/shadow-y")
    p = cfg["config"]["process"][0]
    assert p["model"]["name_or_path"] == "/tmp/shadow-y"
    # the policy still applied (it projects arch only, not name_or_path)
    assert p["train"]["lr"] == 2.5e-5


# --------------------------------------------------------------------------- #
# 7. constants provenance pins (OBSERVED 2026-08-18 header captures)
# --------------------------------------------------------------------------- #

def test_constants_provenance_pins():
    ideo = evalgrid_constants.IDEOGRAM4_FP8_TENSORS
    krea = evalgrid_constants.KREA2_FP8_TENSORS
    assert len(ideo) == 211
    assert len(krea) == 256
    assert evalgrid_constants.IDEOGRAM4_EVAL_SCALE == 1.0
    assert ideo["layers.0.adaln_modulation.weight"] == (18432, 512)
    assert ideo["layers.20.attention.o.weight"] == (4608, 4608)
    s, shape = krea["txtfusion.refiner_blocks.0.attn.wq.weight"]
    assert shape == (2560, 2560)
    assert s == 0.0014648438664153218  # exact f32 scale from the eval header
    assert krea["blocks.0.attn.wq.weight"] == (0.004150390625, (6144, 6144))
    # every krea2 eval scale is a real positive absmax-style scale, none is 1.0
    for name, (scale, sh) in krea.items():
        assert 0.0 < scale < 1.0, name
        assert len(sh) == 2, name
