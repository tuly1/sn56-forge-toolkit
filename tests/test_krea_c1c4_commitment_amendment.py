"""Bind the C1-C4 shape amendment to the already-published commitment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


_REPO = Path(__file__).parents[1]
_CALIBRATION = _REPO / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_decision  # noqa: E402
import krea_c1c4_amendment  # noqa: E402
import krea_execution_plan  # noqa: E402


_PLAN = _REPO / "ops/calibration/week5/krea-discovery-plan.json"
_AMENDMENT = _REPO / ("ops/calibration/week5/krea-c1c4-shape-contract-amendment.json")
_COMMITMENT = "0a12c416bcef48805132e80f9de65d0d248ef4415d617715d5736c189a379dbc"
_PRE_AMENDMENT_PLAN_SHA = (
    "6365f150352de1497fbf32edc8ea07bc2859c3096c95796cff708c89382aee6a"
)
_PRE_AMENDMENT_COMMIT = "1bd7477717ab8d96d208d9fe265f071f08e47e73"
_SHAPES = {
    "C1": {
        "concept_class": "architectural object",
        "training_pairs": 20,
        "evaluation_rows": 6,
    },
    "C2": {
        "concept_class": "art/print-style series",
        "training_pairs": 45,
        "evaluation_rows": 6,
    },
    "C3": {
        "concept_class": "natural subject",
        "training_pairs": 30,
        "evaluation_rows": 8,
    },
    "C4": {
        "concept_class": "product/design object set",
        "training_pairs": 12,
        "evaluation_rows": 5,
    },
}
_MANIFESTS = {
    "C1": "ed287150fd4d189b3a0964d87c5fc50de11851ab372dabe30da9d9f87fdc450e",
    "C2": "902a4a6716a9210694f3f441d54b4def19e9bc64d0a49be4cb832ccff8605083",
    "C3": "74ebbfaf91b156741d34b10ba2d37600076844c010ea6ea83d4af36a386eda09",
    "C4": "7a3fb670bed78d851cf8c066696b61ccc79d78dffd1ecb633520493772210872",
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_public_record(locator: str) -> Path | None:
    for root in (_REPO, *_REPO.parents):
        candidate = root / locator
        if candidate.is_file():
            return candidate
    return None


def _executable_plan() -> dict:
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    plan["status"] = "sealed_executable"
    plan["gpu_execution_authorized"] = True
    plan["gpu_blockers"] = []
    for fixture in ("D1", "D2"):
        plan["discovery_tasks"][fixture]["identity"] = {
            "concept_id": f"concept-{fixture}"
        }
        plan["discovery_tasks"][fixture]["fixture_split_manifest_sha256"] = (
            hashlib.sha256(f"fixture-{fixture}".encode()).hexdigest()
        )
    profiles = plan["budget_contract"]["throughput_profiles_by_equivalence_class"]
    for name in profiles:
        profiles[name] = hashlib.sha256(f"profile-{name}".encode()).hexdigest()
    plan["confirmation_contract"]["identities"] = {
        fixture: hashlib.sha256(f"confirmation-{fixture}".encode()).hexdigest()
        for fixture in ("C1", "C2", "C3", "C4")
    }
    return plan


def test_shape_amendment_self_digest_and_pre_amendment_identity_are_bound():
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    amendment = json.loads(_AMENDMENT.read_text(encoding="utf-8"))

    assert krea_decision.validate_confirmation_shape_amendment(amendment) == amendment
    binding = plan["confirmation_fixture_commitment"]["shape_contract_amendment"]
    assert binding == {
        "path": "ops/calibration/week5/krea-c1c4-shape-contract-amendment.json",
        "file_sha256": _file_sha(_AMENDMENT),
        "amendment_sha256": amendment["amendment_sha256"],
    }
    assert amendment["amends_discovery_plan_file_sha256"] == _PRE_AMENDMENT_PLAN_SHA
    assert amendment["amends_discovery_plan_commit"] == _PRE_AMENDMENT_COMMIT
    original = subprocess.run(
        [
            "git",
            "show",
            f"{_PRE_AMENDMENT_COMMIT}:ops/calibration/week5/krea-discovery-plan.json",
        ],
        cwd=_REPO,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert hashlib.sha256(original).hexdigest() == _PRE_AMENDMENT_PLAN_SHA
    assert "after the public commitment and the independent reviewer finding" in (
        amendment["authorship_order"]
    )


def test_amendment_preserves_commitment_and_replaces_size_aliases_with_exact_shapes():
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    amendment = json.loads(_AMENDMENT.read_text(encoding="utf-8"))
    commitment = plan["confirmation_fixture_commitment"]

    assert plan["confirmation_contract"]["fixture_shape_contract"] == _SHAPES
    assert amendment["amended_fixture_shape_contract"] == _SHAPES
    assert amendment["fixture_commitment_resealed"] is False
    assert amendment["implementation_read_sealed_contents"] is False
    assert (
        commitment["commitment_sha256"]
        == amendment["commitment_sha256_before"]
        == amendment["commitment_sha256_after"]
        == _COMMITMENT
    )
    concatenated = "".join(_MANIFESTS[fixture] for fixture in ("C1", "C2", "C3", "C4"))
    assert hashlib.sha256(concatenated.encode("ascii")).hexdigest() == _COMMITMENT


def test_plan_matches_the_current_published_commitment_when_docs_are_present():
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    commitment = plan["confirmation_fixture_commitment"]
    public_record = _find_public_record(commitment["public_record"])
    if public_record is None:
        pytest.skip("external SN56-project public commitment is not in this checkout")
    raw = public_record.read_bytes()
    text = raw.decode("utf-8")

    assert hashlib.sha256(raw).hexdigest() == commitment["public_record_sha256"]
    published_commitment = re.search(r"COMMITMENT = ([0-9a-f]{64})", text)
    assert published_commitment is not None
    assert published_commitment.group(1) == commitment["commitment_sha256"]
    for fixture, shape in _SHAPES.items():
        inventory = re.search(
            rf"^\| {fixture} \| ([^|]+?) \| (\d+) \| (\d+) \|",
            text,
            flags=re.MULTILINE,
        )
        assert inventory is not None
        assert inventory.groups() == (
            shape["concept_class"],
            str(shape["training_pairs"]),
            str(shape["evaluation_rows"]),
        )
        manifest = re.search(rf"^- {fixture} `([0-9a-f]{{64}})`", text, re.MULTILINE)
        assert manifest is not None
        assert manifest.group(1) == _MANIFESTS[fixture]


@pytest.mark.parametrize(
    "mutation",
    ("reseal", "commitment", "shape", "source", "prior_plan", "self_digest"),
)
def test_shape_amendment_rejects_resealing_or_provenance_drift(mutation: str):
    amendment = json.loads(_AMENDMENT.read_text(encoding="utf-8"))
    bad = deepcopy(amendment)
    if mutation == "reseal":
        bad["fixture_commitment_resealed"] = True
    elif mutation == "commitment":
        bad["commitment_sha256_after"] = "0" * 64
    elif mutation == "shape":
        bad["amended_fixture_shape_contract"]["C4"]["training_pairs"] = 40
    elif mutation == "source":
        bad["source_public_record"] = "/absolute/nonportable/path.md"
    elif mutation == "prior_plan":
        bad["amends_discovery_plan_file_sha256"] = "0" * 64
    else:
        bad["amendment_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        krea_decision.validate_confirmation_shape_amendment(bad)


@pytest.mark.parametrize("failure", ("missing", "corrupt", "hash_drift"))
def test_policy_load_fails_closed_when_repo_amendment_is_unusable(
    monkeypatch, tmp_path, failure: str
):
    relative = Path(krea_c1c4_amendment.AMENDMENT_PATH)
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    if failure == "corrupt":
        target.write_bytes(b"{not-json}\n")
    elif failure == "hash_drift":
        target.write_bytes(_AMENDMENT.read_bytes() + b"\n")
    monkeypatch.setattr(krea_c1c4_amendment, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(ValueError):
        krea_decision._validate_discovery_plan(_executable_plan())


def test_execution_discovery_load_invokes_repo_amendment_validator(
    monkeypatch, tmp_path
):
    discovery = json.loads(_PLAN.read_text(encoding="utf-8"))
    local = tmp_path / "discovery.json"
    local.write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")

    def reject(_value):
        raise RuntimeError("repo amendment validator reached")

    monkeypatch.setattr(
        krea_execution_plan.krea_c1c4_amendment,
        "validate_bound_plan_amendment",
        reject,
    )
    with pytest.raises(RuntimeError, match="validator reached"):
        krea_execution_plan.validate_discovery_semantics(
            {"path": str(local), "sha256": _file_sha(local)},
            arm_id="K1",
            fixture_id="D1",
            fixture_manifest_sha256="1" * 64,
            training_pair_count=24,
            seed_role="A",
            seed=42_565_431,
            throughput_equivalence_class="unused",
            execution_recipe={"fields": {}},
            schedule_mode="measured_budget_fill",
            predeclared_recipe_axes=[],
            basis_mode="internal",
        )
