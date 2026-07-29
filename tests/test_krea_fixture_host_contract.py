"""Adversarial tests for the Week-5 fixture and measured-host gates."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _CALIBRATION / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = _load("krea_provenance")
sys.modules["krea_provenance"] = provenance
dataset_identity = _load("krea_dataset_identity")
sys.modules["krea_dataset_identity"] = dataset_identity
fixture = _load("krea_fixture")
host = _load("krea_host_identity")


def _utc(seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("role", "training", "evaluation"),
    [
        ("C1", 20, 6),
        ("C2", 45, 6),
        ("C3", 30, 8),
        ("C4", 12, 5),
    ],
)
def test_confirmation_fixture_roles_enforce_published_exact_shapes(
    role, training, evaluation
):
    fixture._validate_role_counts(role, training, evaluation)


@pytest.mark.parametrize(
    ("role", "training", "evaluation"),
    [
        ("C1", 19, 6),
        ("C1", 20, 24),
        ("C2", 20, 6),
        ("C2", 45, 24),
        ("C3", 40, 8),
        ("C3", 30, 40),
        ("C4", 40, 5),
        ("C4", 12, 40),
        ("C4", True, 5),
    ],
)
def test_confirmation_fixture_roles_reject_generalized_or_wrong_counts(
    role, training, evaluation
):
    with pytest.raises(ValueError):
        fixture._validate_role_counts(role, training, evaluation)


@pytest.mark.parametrize(
    ("role", "training", "evaluation"),
    [
        ("D1", 18, 24),
        ("D1", 24, 24),
        ("D2", 36, 40),
        ("D2", 48, 40),
    ],
)
def test_discovery_fixture_ranges_remain_distinct_and_unchanged(
    role, training, evaluation
):
    fixture._validate_role_counts(role, training, evaluation)


def _row(role: str, index: int) -> dict:
    token = f"{role}-{index}"
    group = {
        field: f"{field}-{token}"
        for field in (
            "source_id",
            "creator_id",
            "burst_id",
            "scene_id",
            "play_root_id",
            "human_similarity_cluster_id",
        )
    }
    if role == "D2":
        group.update(
            {
                "play_component_id": f"play-component-{token}",
                "accession_family_id": f"accession-family-{token}",
            }
        )
    return {
        "row_id": f"row-{token}",
        "content_sha256": _sha(f"content-{token}"),
        "image_sha256": _sha(f"image-{token}"),
        "decoded_pixels_sha256": _sha(f"pixels-{token}"),
        "normalized_caption_sha256": _sha(f"caption-{token}"),
        "perceptual_hash64": f"{1 + list(fixture._CROSS_FIXTURE_ROLES).index(role) * 2 + index:016x}",
        "group_identity": group,
    }


def _fixtures() -> list[dict]:
    prepared_at = _utc(-120)
    reviewed_at = _utc(-90)
    values = []
    for role in fixture._CROSS_FIXTURE_ROLES:
        values.append(
            {
                "concept_id": f"concept-{role}",
                "experimental_role": role,
                "trigger_token": f"trigger-{role}",
                "manifest_sha256": _sha(f"manifest-{role}"),
                "preparer_identity": f"Preparer {role}",
                "source_rights": {
                    "retrieved_at_utc": prepared_at,
                    "reviewed_at_utc": reviewed_at,
                    "reviewer_identity": f"Rights {role}",
                },
                "caption_policy": {
                    "reviewed_at_utc": reviewed_at,
                    "reviewer_identity": f"Caption {role}",
                },
                "near_duplicate_policy": {
                    "maximum_hamming_distance": 0,
                    "group_disjoint_fields": sorted(
                        {
                            "burst_id",
                            "human_similarity_cluster_id",
                            "play_root_id",
                            "scene_id",
                        }
                        | (
                            {"play_component_id", "accession_family_id"}
                            if role == "D2"
                            else set()
                        )
                    ),
                    "human_similarity_review": {
                        "reviewed_at_utc": reviewed_at,
                        "reviewer_identity": f"Similarity {role}",
                    },
                },
                "training_rows": [_row(role, 0)],
                "evaluation_rows": [_row(role, 1)],
            }
        )
    return values


def test_d2_group_identity_requires_component_and_accession_family() -> None:
    group = _row("D2", 0)["group_identity"]
    assert fixture._normalize_group_identity(
        group, role="D2", label="D2 group"
    ) == dict(sorted(group.items()))

    for missing in ("play_component_id", "accession_family_id"):
        incomplete = {key: value for key, value in group.items() if key != missing}
        with pytest.raises(ValueError, match="keys mismatch"):
            fixture._normalize_group_identity(incomplete, role="D2", label="D2 group")

    with pytest.raises(ValueError, match="keys mismatch"):
        fixture._normalize_group_identity(group, role="D1", label="D1 group")


@pytest.mark.parametrize("shared_field", ("play_component_id", "accession_family_id"))
def test_d2_component_and_accession_groups_are_leakage_boundaries(shared_field) -> None:
    training = _row("D2", 0)
    evaluation = _row("D2", 1)
    evaluation["group_identity"][shared_field] = training["group_identity"][
        shared_field
    ]
    group_fields = sorted(
        fixture._BASE_GROUP_DISJOINT_FIELDS | fixture._D2_GROUP_FIELDS
    )

    report = fixture._duplicates(
        [training],
        [evaluation],
        threshold=0,
        group_disjoint_fields=tuple(group_fields),
    )

    assert [match["field"] for match in report["cross_split_group_matches"]] == [
        shared_field
    ]


@pytest.mark.parametrize("missing", ("play_component_id", "accession_family_id"))
def test_d2_disjointness_policy_cannot_omit_new_group_boundaries(missing) -> None:
    fields = sorted(
        (fixture._BASE_GROUP_DISJOINT_FIELDS | fixture._D2_GROUP_FIELDS) - {missing}
    )
    with pytest.raises(ValueError, match="leakage policy"):
        fixture._normalize_group_disjoint_fields(fields, role="D2")


def test_fixture_decode_applies_exif_orientation_before_rgb_identity(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    pil_image_ops = pytest.importorskip("PIL.ImageOps")
    image_path = tmp_path / "oriented.jpg"
    caption_path = tmp_path / "oriented.txt"
    image = pil_image.new("RGB", (2, 3), color=(25, 75, 125))
    exif = pil_image.Exif()
    exif[274] = 6
    image.save(image_path, format="JPEG", exif=exif)
    caption_bytes = b"a reviewed fixture caption\n"
    caption_path.write_bytes(caption_bytes)
    image_bytes = image_path.read_bytes()
    evaluator_row = {
        "image": image_path.name,
        "prompt": caption_path.name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(caption_bytes).hexdigest(),
        "image_format": "JPEG",
    }
    group = _row("D1", 0)["group_identity"]

    decoded = fixture._decode_row(tmp_path, evaluator_row, group)

    with pil_image.open(image_path) as opened:
        assert opened.size == (2, 3)
        expected = pil_image_ops.exif_transpose(opened).convert("RGB")
        assert expected.size == (3, 2)
        expected_pixels = expected.tobytes()
    assert (decoded["width"], decoded["height"]) == (3, 2)
    assert (
        decoded["decoded_pixels_sha256"] == hashlib.sha256(expected_pixels).hexdigest()
    )


def _cross_review(fixtures: list[dict]) -> dict:
    return {
        "schema": 1,
        "kind": "forge-krea-cross-fixture-human-similarity-review",
        "fixture_manifest_sha256s": {
            item["experimental_role"]: item["manifest_sha256"]
            for item in sorted(fixtures, key=lambda row: row["experimental_role"])
        },
        "reviewer_identity": "Independent Reviewer",
        "reviewed_at_utc": _utc(-30),
        "decision": "passed",
        "reviewed_pairs": fixture._cross_fixture_pairs(fixtures),
        "flagged_pairs": [],
        "claim_limit": "cross-fixture-nonoverlap-only",
    }


def test_cross_fixture_review_is_exhaustive_portable_and_consumer_bound(
    tmp_path, monkeypatch
):
    fixtures = _fixtures()
    monkeypatch.setattr(fixture, "validate_manifest", lambda value: value)
    review = _cross_review(fixtures)
    review_path = tmp_path / "cross-review.json"
    review_path.write_bytes(provenance.canonical_bytes(review) + b"\n")

    result = fixture.cross_fixture_disjoint(fixtures, human_review_record=review_path)

    assert result["fixture_manifest_sha256s"] == review["fixture_manifest_sha256s"]
    assert result["reviewed_pair_count"] == 60
    assert result["reviewer_identity_assurance"].endswith(
        "not-cryptographic-authentication"
    )
    portable = {
        key: value for key, value in result.items() if key not in {"path", "sha256"}
    }
    fixture.validate_cross_fixture_binding(
        portable,
        fixtures=fixtures,
        review_record=review,
        review_file_sha256=result["review_file_sha256"],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_pair",
        "reordered_pairs",
        "wrong_manifest",
        "early",
        "future",
        "same_human",
    ],
)
def test_cross_fixture_review_rejects_partial_or_nonindependent_evidence(
    mutation, monkeypatch
):
    fixtures = _fixtures()
    monkeypatch.setattr(fixture, "validate_manifest", lambda value: value)
    review = _cross_review(fixtures)
    if mutation == "missing_pair":
        review["reviewed_pairs"] = review["reviewed_pairs"][:-1]
    elif mutation == "reordered_pairs":
        review["reviewed_pairs"][0], review["reviewed_pairs"][1] = (
            review["reviewed_pairs"][1],
            review["reviewed_pairs"][0],
        )
    elif mutation == "wrong_manifest":
        review["fixture_manifest_sha256s"]["C4"] = _sha("wrong")
    elif mutation == "early":
        review["reviewed_at_utc"] = _utc(-180)
    elif mutation == "future":
        review["reviewed_at_utc"] = _utc(120)
    else:
        review["reviewer_identity"] = "  caption   c1  "
    with pytest.raises(ValueError):
        fixture.validate_cross_fixture_review(review, fixtures=fixtures)


def test_cross_fixture_binding_rejects_digest_tampering(monkeypatch):
    fixtures = _fixtures()
    monkeypatch.setattr(fixture, "validate_manifest", lambda value: value)
    review = _cross_review(fixtures)
    file_sha = _sha("review-file")
    body = fixture._cross_fixture_binding_body(review, review_file_sha256=file_sha)
    binding = {**body, "binding_sha256": provenance.canonical_sha256(body)}
    binding["reviewed_pair_count"] -= 1
    with pytest.raises(ValueError, match="exact review"):
        fixture.validate_cross_fixture_binding(
            binding,
            fixtures=fixtures,
            review_record=review,
            review_file_sha256=file_sha,
        )


def _host_manifest() -> dict:
    gib = 1024**3
    body = {
        "schema": 2,
        "kind": "forge-krea-host-execution-identity",
        "static": {
            "instance": {
                "machine_id_sha256": _sha("machine"),
                "product_uuid_sha256": _sha("product"),
                "boot_id_sha256": _sha("boot"),
                "cgroup_v2_path": "/system.slice/forge.scope",
                "assurance": host._INSTANCE_ASSURANCE,
            },
            "cpu": {
                "model": "Test CPU",
                "logical_cpus": 16,
                "process_affinity_cpu_ids": list(range(8)),
                "cgroup_cpuset_cpu_ids": list(range(2, 8)),
                "effective_cpu_ids": list(range(2, 8)),
                "allowed_logical_cpus": 6,
                "cgroup_cpuset_logical_cpus": 6,
                "cpu_quota_cores": 4.0,
                "effective_cpu_capacity": 4.0,
            },
            "memory": {
                "total_bytes": 64 * gib,
                "cgroup_limit_bytes": 32 * gib,
                "effective_capacity_bytes": 32 * gib,
            },
            "checkpoint_filesystem": {
                "checkpoint_path": "/checkpoints",
                "mount_target": "/",
                "source": "/dev/nvme0n1p1",
                "filesystem_type": "ext4",
                "mount_options": ["relatime", "rw"],
                "device_major_minor": "259:1",
                "device_id": 66305,
            },
            "gpu": {
                "uuid": "GPU-test",
                "name": "NVIDIA H100 PCIe",
                "driver_version": "570.01",
                "mig_mode": "Disabled",
                "power_limit_w": 350.0,
                "max_sm_clock_mhz": 1755,
                "max_memory_clock_mhz": 1593,
                "total_memory_mib": 81920,
            },
        },
        "preflight_policy": {
            "maximum_load_per_effective_cpu": 0.5,
            "minimum_available_memory_bytes": 8 * gib,
            "minimum_checkpoint_free_bytes": 4 * gib,
            "maximum_gpu_utilization_percent": 5.0,
            "minimum_free_gpu_memory_mib": 70000,
            "maximum_foreign_compute_processes": 0,
            "storage_probe_bytes": 4 * 1024 * 1024,
            "minimum_checkpoint_write_mib_s": 2.0,
            "minimum_checkpoint_read_mib_s": 2.0,
            "maximum_checkpoint_fsync_s": 2.0,
            "storage_probe_tool_sha256": host._module_sha256(),
        },
    }
    return {
        **body,
        "host_execution_identity_sha256": host.canonical_sha256(body),
    }


def _live(manifest: dict) -> dict:
    gib = 1024**3
    observed_at = _utc()
    return {
        "static": deepcopy(manifest["static"]),
        "live": {
            "observed_at_utc": observed_at,
            "load_1m": 1.0,
            "host_available_memory_bytes": 40 * gib,
            "cgroup_available_memory_bytes": 20 * gib,
            "available_memory_bytes": 20 * gib,
            "checkpoint_free_bytes": 10 * gib,
            "gpu_utilization_percent": 0.0,
            "free_gpu_memory_mib": 80000,
            "compute_process_pids": [],
            "checkpoint_storage_probe": {
                "checkpoint_path": "/checkpoints",
                "device_major_minor": "259:1",
                "bytes": 4 * 1024 * 1024,
                "content_sha256": host._storage_probe_content_sha256(4 * 1024 * 1024),
                "write_s": 1.0,
                "fsync_s": 0.5,
                "read_s": 1.0,
                "write_mib_s": 4.0,
                "read_mib_s": 4.0,
                "cache_drop_requested": True,
                "tool_sha256": host._module_sha256(),
                "observed_at_utc": observed_at,
            },
        },
    }


def _reseal(manifest: dict) -> dict:
    body = {
        key: value
        for key, value in manifest.items()
        if key != "host_execution_identity_sha256"
    }
    manifest["host_execution_identity_sha256"] = host.canonical_sha256(body)
    return manifest


def test_host_manifest_and_live_preflight_accept_complete_bound_observation(
    monkeypatch,
):
    manifest = _host_manifest()
    observed = _live(manifest)
    monkeypatch.setattr(host, "observe", lambda *args, **kwargs: observed)
    assert host.validate_manifest(manifest) == manifest
    assert host.verify_live(manifest, checkpoint_path=Path("/checkpoints")) == observed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_cpu_capacity", 5.0),
        ("logical_cpus", 0),
        ("cpu_quota_cores", float("nan")),
    ],
)
def test_host_manifest_rejects_invalid_cpu_capacity(field, value):
    manifest = _host_manifest()
    manifest["static"]["cpu"][field] = value
    if field != "cpu_quota_cores":
        _reseal(manifest)
    with pytest.raises(ValueError):
        host.validate_manifest(manifest)


def test_host_manifest_rejects_read_only_or_unsafe_path_contract():
    manifest = _host_manifest()
    manifest["static"]["checkpoint_filesystem"]["mount_options"] = ["ro"]
    _reseal(manifest)
    with pytest.raises(ValueError, match="read-write"):
        host.validate_manifest(manifest)
    manifest = _host_manifest()
    manifest["static"]["checkpoint_filesystem"]["checkpoint_path"] = "relative"
    _reseal(manifest)
    with pytest.raises(ValueError, match="absolute"):
        host.validate_manifest(manifest)


def test_host_manifest_rejects_cpu_set_or_instance_assurance_tampering():
    manifest = _host_manifest()
    manifest["static"]["cpu"]["effective_cpu_ids"] = [2, 3]
    manifest["static"]["cpu"]["allowed_logical_cpus"] = 2
    manifest["static"]["cpu"]["effective_cpu_capacity"] = 2.0
    _reseal(manifest)
    with pytest.raises(ValueError, match="intersection"):
        host.validate_manifest(manifest)
    manifest = _host_manifest()
    manifest["static"]["instance"]["assurance"] = "cryptographically-attested"
    _reseal(manifest)
    with pytest.raises(ValueError, match="overstates"):
        host.validate_manifest(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "instance",
        "memory_math",
        "gpu_nan",
        "gpu_free_impossible",
        "pids_duplicate",
        "probe_path",
        "probe_bytes",
        "probe_digest",
        "probe_rate",
        "stale",
    ],
)
def test_host_live_preflight_rejects_tampered_observation(mutation, monkeypatch):
    manifest = _host_manifest()
    observed = _live(manifest)
    live = observed["live"]
    probe = live["checkpoint_storage_probe"]
    if mutation == "instance":
        observed["static"]["instance"]["boot_id_sha256"] = _sha("other-boot")
    elif mutation == "memory_math":
        live["available_memory_bytes"] += 1
    elif mutation == "gpu_nan":
        live["gpu_utilization_percent"] = float("nan")
    elif mutation == "gpu_free_impossible":
        live["free_gpu_memory_mib"] = 90000
    elif mutation == "pids_duplicate":
        live["compute_process_pids"] = [42, 42]
    elif mutation == "probe_path":
        probe["checkpoint_path"] = "/other"
    elif mutation == "probe_bytes":
        probe["bytes"] += 1
    elif mutation == "probe_digest":
        probe["content_sha256"] = "bad"
    elif mutation == "probe_rate":
        probe["write_mib_s"] = 99.0
    else:
        stale = _utc(-600)
        live["observed_at_utc"] = stale
        probe["observed_at_utc"] = stale
    monkeypatch.setattr(host, "observe", lambda *args, **kwargs: observed)
    with pytest.raises((ValueError, RuntimeError)):
        host.verify_live(manifest, checkpoint_path=Path("/checkpoints"))


@pytest.mark.parametrize(
    "mutation",
    [
        "load",
        "memory",
        "disk",
        "gpu_busy",
        "gpu_memory",
        "foreign",
        "write",
        "read",
        "fsync",
    ],
)
def test_host_live_thresholds_fail_closed(mutation, monkeypatch):
    manifest = _host_manifest()
    observed = _live(manifest)
    live = observed["live"]
    probe = live["checkpoint_storage_probe"]
    if mutation == "load":
        live["load_1m"] = 3.0
    elif mutation == "memory":
        live["cgroup_available_memory_bytes"] = 1
        live["available_memory_bytes"] = 1
    elif mutation == "disk":
        live["checkpoint_free_bytes"] = 1
    elif mutation == "gpu_busy":
        live["gpu_utilization_percent"] = 6.0
    elif mutation == "gpu_memory":
        live["free_gpu_memory_mib"] = 69999
    elif mutation == "foreign":
        live["compute_process_pids"] = [4242]
    elif mutation == "write":
        probe["write_s"] = 4.0
        probe["write_mib_s"] = 1.0
    elif mutation == "read":
        probe["read_s"] = 4.0
        probe["read_mib_s"] = 1.0
    else:
        probe["fsync_s"] = 3.0
    monkeypatch.setattr(host, "observe", lambda *args, **kwargs: observed)
    with pytest.raises(RuntimeError, match="thresholds failed"):
        host.verify_live(manifest, checkpoint_path=Path("/checkpoints"))


def test_effective_cgroup_constraints_include_all_ancestors(tmp_path, monkeypatch):
    base = tmp_path / "cgroup"
    child = base / "slice"
    leaf = child / "scope"
    leaf.mkdir(parents=True)
    rows = [
        (base, "150000 100000", str(40 * 1024**3), str(35 * 1024**3)),
        (child, "400000 100000", str(32 * 1024**3), str(8 * 1024**3)),
        (leaf, "max 100000", "max", str(2 * 1024**3)),
    ]
    for path, cpu_max, memory_max, memory_current in rows:
        (path / "cpu.max").write_text(cpu_max, encoding="ascii")
        (path / "memory.max").write_text(memory_max, encoding="ascii")
        (path / "memory.current").write_text(memory_current, encoding="ascii")
    (leaf / "cpuset.cpus.effective").write_text("0-3,8-9", encoding="ascii")
    monkeypatch.setattr(host, "_cgroup_base", lambda: base)
    monkeypatch.setattr(host, "_cgroup_root", lambda: leaf)

    constraints = host._cgroup_constraints()

    assert constraints["path"] == "/slice/scope"
    assert constraints["cpuset_cpu_ids"] == [0, 1, 2, 3, 8, 9]
    assert constraints["cpuset_logical_cpus"] == 6
    assert constraints["cpu_quota_cores"] == 1.5
    assert constraints["memory_limit_bytes"] == 32 * 1024**3
    assert constraints["memory_available_bytes"] == 5 * 1024**3


def test_checkpoint_path_rejects_relative_file_and_symlink_ancestors(tmp_path):
    directory = tmp_path / "real"
    directory.mkdir()
    assert host._safe_checkpoint_path(directory) == directory
    with pytest.raises(ValueError, match="absolute"):
        host._safe_checkpoint_path(Path("relative"))
    file_path = directory / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        host._safe_checkpoint_path(file_path)
    linked = tmp_path / "linked"
    linked.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestor"):
        host._safe_checkpoint_path(linked)


def test_storage_probe_checks_bytes_and_leaves_no_probe_file(tmp_path):
    result = host._storage_probe(tmp_path, byte_count=4 * 1024 * 1024)
    assert result["bytes"] == 4 * 1024 * 1024
    assert result["write_mib_s"] > 0
    assert result["read_mib_s"] > 0
    assert not list(tmp_path.glob(".krea-io-probe-*"))
