"""CPU-only contracts for the Week-7 procedural HKE candidate renderer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops"
    / "experiments"
    / "week7"
    / "hke_procedural_renderer.py"
)
SPEC = importlib.util.spec_from_file_location("hke_procedural_renderer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

DISCOVERY_KEY = b"week7-discovery-key-material-0001"
CONFIRMATION_KEY = b"week7-confirm-key-material-0000002"
GENERATOR_COMMIT = "a" * 40
GENERATOR_TREE = "b" * 40
RIGHTS = {
    "author_record": "SN56 first-party procedural renderer",
    "rights_owner": "SN56 project owner",
    "license_or_use_grant": "first-party training and evaluation fixture use",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _tree_identity(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_semantic(path: Path, value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "semantic_sha256"}
    value["semantic_sha256"] = renderer.semantic_sha256(body)
    path.write_bytes(renderer.canonical_bytes(value) + b"\n")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("week7-hke-renderer")
    public = root / "public-boundary" / "public"
    public.parent.mkdir()
    custodian = root / "custodian"
    result = renderer.build_candidate(
        public_output=public,
        custodian_output=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        public_boundary_roots=(public.parent,),
        **RIGHTS,
    )
    return public, custodian, result


def test_exact_shapes_candidate_governance_and_row_evidence(built) -> None:
    public, custodian, result = built
    assert result["status"] == "candidate_unreviewed"
    assert result["governance"] == {
        "human_review": "not_performed",
        "agent_is_not_human": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    expected = {
        "W7-HKE-SOCIAL-A": {
            "discovery": {"D1": (10, 8), "D2": (10, 8)},
            "confirmation": {"C1": (10, 8), "C2": (10, 8)},
        },
        "W7-HKE-PRODUCT-A": {
            "discovery": {"D1": (10, 8)},
            "confirmation": {"C1": (10, 8)},
        },
        "W7-HKE-LOGO-UI-A": {
            "discovery": {"D1": (10, 8)},
            "confirmation": {"C1": (10, 8)},
        },
    }
    confirmation = _json(custodian / "CONFIRMATION-MANIFEST.json")
    total = 0
    phase_totals = {"discovery": 0, "confirmation": 0}
    for fixture_id, expected_phases in expected.items():
        public_family = result["families"][fixture_id]
        private_family = confirmation["families"][fixture_id]
        discovery_count = sum(
            training + evaluation
            for training, evaluation in expected_phases["discovery"].values()
        )
        confirmation_count = sum(
            training + evaluation
            for training, evaluation in expected_phases["confirmation"].values()
        )
        assert public_family["discovery"]["row_count"] == discovery_count
        assert len(public_family["discovery"]["rows"]) == discovery_count
        assert public_family["confirmation"]["row_count"] == confirmation_count
        assert private_family["row_count"] == confirmation_count
        phase_totals["discovery"] += discovery_count
        phase_totals["confirmation"] += confirmation_count
        total += discovery_count + confirmation_count

        for phase, family_record in (
            ("discovery", public_family["discovery"]),
            ("confirmation", private_family),
        ):
            assert set(family_record["packs"]) == set(expected_phases[phase])
            flattened: list[dict] = []
            for pack_name, (training_count, evaluation_count) in expected_phases[
                phase
            ].items():
                pack = family_record["packs"][pack_name]
                assert pack["row_count"] == training_count + evaluation_count
                assert set(pack["splits"]) == {"training", "evaluation"}
                assert pack["splits"]["training"]["row_count"] == training_count
                assert pack["splits"]["evaluation"]["row_count"] == evaluation_count
                assert len(pack["splits"]["training"]["rows"]) == training_count
                assert len(pack["splits"]["evaluation"]["rows"]) == evaluation_count
                assert pack["rows"] == (
                    pack["splits"]["training"]["rows"]
                    + pack["splits"]["evaluation"]["rows"]
                )
                assert all(row["pack"] == pack_name for row in pack["rows"])
                assert all(
                    row["split_role"] == "training"
                    for row in pack["splits"]["training"]["rows"]
                )
                assert all(
                    row["split_role"] == "evaluation"
                    for row in pack["splits"]["evaluation"]["rows"]
                )
                flattened.extend(pack["rows"])
            assert flattened == family_record["rows"]

        for row in public_family["discovery"]["rows"] + private_family["rows"]:
            assert row["split_role"] in {"training", "evaluation"}
            assert row["parameters"]["split_role"] == row["split_role"]
            assert row["parameters"]["pack"] == row["pack"]
            assert row["parameters_sha256"] == renderer.semantic_sha256(
                row["parameters"]
            )
            assert row["group_identity_sha256"] == renderer.semantic_sha256(
                row["group_identity"]
            )
            assert row["rights_declaration"] == renderer.RIGHTS_DECLARATION
            assert row["visible_glyph_transcript"]

        public_confirmation_packs = public_family["confirmation"]["packs"]
        for pack_name, (training_count, evaluation_count) in expected_phases[
            "confirmation"
        ].items():
            assert public_confirmation_packs[pack_name]["row_count"] == (
                training_count + evaluation_count
            )
            assert (
                public_confirmation_packs[pack_name]["training_row_count"]
                == training_count
            )
            assert (
                public_confirmation_packs[pack_name]["evaluation_row_count"]
                == evaluation_count
            )
            assert set(public_confirmation_packs[pack_name]) == {
                "row_count",
                "training_row_count",
                "evaluation_row_count",
                "semantic_commitment_sha256",
            }

    assert phase_totals == {"discovery": 72, "confirmation": 72}
    assert total == 144
    assert result["cross_candidate_evidence"]["row_count"] == 144
    assert result["cross_candidate_evidence"]["pair_comparisons"] == 10296
    assert result["cross_candidate_evidence"]["exact_image_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["decoded_pixel_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["normalized_caption_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["group_identity_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["perceptual_near_duplicate_count"] == 0
    assert result["rights_record"]["rights_owner"] == RIGHTS["rights_owner"]
    assert (
        renderer.verify_candidate(
            public, custodian, public_boundary_roots=(public.parent,)
        )["verified_rows"]
        == 144
    )


def test_confirmation_is_custodian_only_and_public_manifest_leaks_no_membership(
    built,
) -> None:
    public, custodian, result = built
    public_bytes = (public / "CANDIDATE-MANIFEST.json").read_bytes()
    confirmation = _json(custodian / "CONFIRMATION-MANIFEST.json")
    assert not (public / "confirmation").exists()
    assert (custodian / "confirmation").is_dir()
    assert str(custodian).encode() not in public_bytes
    assert result["confirmation"]["public_membership_disclosed"] is False
    for family in confirmation["families"].values():
        for row in family["rows"]:
            caption = (
                custodian / "confirmation" / row["relative_caption_path"]
            ).read_bytes()
            for private_value in (
                row["row_id"],
                row["relative_image_path"],
                row["relative_caption_path"],
                row["image_sha256"],
                row["caption_sha256"],
                row["parameters_sha256"],
                row["row_record_sha256"],
                caption.decode("utf-8").strip(),
            ):
                assert private_value.encode("ascii") not in public_bytes


def test_fresh_build_is_byte_deterministic_and_replay_passes(tmp_path: Path) -> None:
    first_public, first_custodian = tmp_path / "b1" / "p1", tmp_path / "c1"
    second_public, second_custodian = tmp_path / "b2" / "p2", tmp_path / "c2"
    first_public.parent.mkdir()
    second_public.parent.mkdir()
    first = renderer.build_candidate(
        public_output=first_public,
        custodian_output=first_custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        public_boundary_roots=(first_public.parent,),
        **RIGHTS,
    )
    second = renderer.build_candidate(
        public_output=second_public,
        custodian_output=second_custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        public_boundary_roots=(second_public.parent,),
        **RIGHTS,
    )
    assert first == second
    assert _tree_identity(first_public) == _tree_identity(second_public)
    assert _tree_identity(first_custodian) == _tree_identity(second_custodian)
    replay = renderer.verify_replay(
        public_output=first_public,
        custodian_output=first_custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        public_boundary_roots=(first_public.parent,),
    )
    assert replay == {
        "status": "PASS",
        "verified_rows": 144,
        "candidate_semantic_sha256": first["semantic_sha256"],
        "dedup_semantic_sha256": first["cross_candidate_evidence"]["semantic_sha256"],
        "discovery_key_commitment_sha256": renderer._phase_key_commitment(
            DISCOVERY_KEY, renderer.DISCOVERY_DOMAIN
        ),
        "confirmation_key_commitment_sha256": renderer._phase_key_commitment(
            CONFIRMATION_KEY, renderer.CONFIRMATION_DOMAIN
        ),
        "contract_path": renderer.DECLARATIVE_CONTRACT_SOURCE_PATH,
        "contract_source_sha256": renderer._contract_source_sha256(),
    }


def test_keys_and_domains_are_distinct_and_wrong_key_fails_replay(
    built, tmp_path: Path, monkeypatch
) -> None:
    public, custodian, _ = built
    fixture = renderer.FIXTURE_CONTRACT[0]
    discovery_image, _, discovery_row = renderer._render(
        fixture, "discovery", 0, DISCOVERY_KEY
    )
    confirmation_image, _, confirmation_row = renderer._render(
        fixture, "confirmation", 0, CONFIRMATION_KEY
    )
    assert discovery_row["parameters"] != confirmation_row["parameters"]
    assert discovery_image != confirmation_image
    assert (
        discovery_row["seed_commitment_sha256"]
        != confirmation_row["seed_commitment_sha256"]
    )
    with pytest.raises(
        renderer.FixtureError, match="manifest identity|replay mismatch"
    ):
        renderer.verify_replay(
            public_output=public,
            custodian_output=custodian,
            discovery_key=b"wrong-discovery-key-material-0000",
            confirmation_key=CONFIRMATION_KEY,
            public_boundary_roots=(public.parent,),
        )
    with pytest.raises(renderer.FixtureError, match="must be distinct"):
        renderer.verify_replay(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=DISCOVERY_KEY,
            public_boundary_roots=(public.parent,),
        )
    with pytest.raises(renderer.FixtureError, match="must be distinct"):
        renderer.build_candidate(
            public_output=public.parent / "same-key-boundary" / "same-key-public",
            custodian_output=public.parent.parent / "same-key-custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=DISCOVERY_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(public.parent / "same-key-boundary",),
            **RIGHTS,
        )
    with monkeypatch.context() as context:
        context.setattr(renderer, "_require_distinct_phase_keys", lambda *_: None)
        (tmp_path / "same-key-boundary").mkdir()
        renderer.build_candidate(
            public_output=tmp_path / "same-key-boundary" / "public",
            custodian_output=tmp_path / "same-key-internally-consistent-private",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=DISCOVERY_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(tmp_path / "same-key-boundary",),
            **RIGHTS,
        )
    with pytest.raises(renderer.FixtureError, match="must be distinct"):
        renderer.verify_replay(
            public_output=tmp_path / "same-key-boundary" / "public",
            custodian_output=tmp_path / "same-key-internally-consistent-private",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=DISCOVERY_KEY,
            public_boundary_roots=(tmp_path / "same-key-boundary",),
        )


def test_swapped_or_mismatched_phase_commitment_fails_replay(
    built, tmp_path: Path
) -> None:
    public, custodian, _ = built
    copied_public = tmp_path / "commitment-boundary" / "commitment-public"
    copied_custodian = tmp_path / "commitment-custodian"
    copied_public.parent.mkdir()
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, copied_custodian)
    discovery_path = copied_public / "DISCOVERY-MANIFEST.json"
    discovery = _json(discovery_path)
    discovery["phase_key_commitment_sha256"] = renderer._phase_key_commitment(
        CONFIRMATION_KEY, renderer.CONFIRMATION_DOMAIN
    )
    _rewrite_semantic(discovery_path, discovery)
    candidate_path = copied_public / "CANDIDATE-MANIFEST.json"
    candidate = _json(candidate_path)
    candidate["discovery_manifest_file_sha256"] = hashlib.sha256(
        discovery_path.read_bytes()
    ).hexdigest()
    _rewrite_semantic(candidate_path, candidate)
    with pytest.raises(renderer.FixtureError, match="discovery manifest identity"):
        renderer.verify_replay(
            public_output=copied_public,
            custodian_output=copied_custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            public_boundary_roots=(copied_public.parent,),
        )


def test_stable_group_dedup_ignores_phase_split_role_and_private_membership() -> None:
    fixture = renderer.FIXTURE_CONTRACT[0]
    discovery_image, _, discovery = renderer._render(
        fixture, "discovery", 0, DISCOVERY_KEY
    )
    confirmation_image, _, confirmation = renderer._render(
        fixture, "confirmation", 0, CONFIRMATION_KEY
    )
    assert discovery["row_id"] != confirmation["row_id"]
    assert discovery["phase"] != confirmation["phase"]
    assert set(discovery["group_identity"]) == {
        "concept_identity",
        "family",
        "layout_semantics",
    }
    serialized_group = renderer.canonical_bytes(discovery["group_identity"])
    for forbidden in (
        b"phase",
        b"pack",
        b"split_role",
        b"ordinal",
        b"membership",
        b"private",
    ):
        assert forbidden not in serialized_group
    confirmation["group_identity"] = discovery["group_identity"]
    confirmation["group_identity_sha256"] = discovery["group_identity_sha256"]
    evidence = renderer._dedup_evidence(
        [(discovery, discovery_image), (confirmation, confirmation_image)]
    )
    assert evidence["group_identity_duplicate_count"] == 1


def test_train_eval_semantic_identity_transplant_is_detected() -> None:
    fixture = renderer.FIXTURE_CONTRACT[0]
    training_image, _, training = renderer._render(
        fixture, "discovery", 0, DISCOVERY_KEY
    )
    evaluation_image, _, evaluation = renderer._render(
        fixture, "discovery", 10, DISCOVERY_KEY
    )
    assert training["pack"] == evaluation["pack"] == "D1"
    assert training["split_role"] == "training"
    assert evaluation["split_role"] == "evaluation"
    assert training["group_identity_sha256"] != evaluation["group_identity_sha256"]

    evaluation["group_identity"] = training["group_identity"]
    evaluation["group_identity_sha256"] = training["group_identity_sha256"]
    evidence = renderer._dedup_evidence(
        [(training, training_image), (evaluation, evaluation_image)]
    )
    assert evidence["group_identity_duplicate_count"] == 1


def test_schema_v2_candidate_is_explicitly_invalidated(built, tmp_path: Path) -> None:
    public, custodian, _ = built
    copied_public = tmp_path / "schema-boundary" / "schema-public"
    copied_custodian = tmp_path / "schema-custodian"
    copied_public.parent.mkdir()
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, copied_custodian)
    candidate_path = copied_public / "CANDIDATE-MANIFEST.json"
    candidate = _json(candidate_path)
    candidate["schema"] = 2
    _rewrite_semantic(candidate_path, candidate)
    with pytest.raises(renderer.FixtureError, match="identity or provenance"):
        renderer.verify_candidate(
            copied_public,
            copied_custodian,
            public_boundary_roots=(copied_public.parent,),
        )


def test_confirmation_membership_caption_and_visible_token_require_private_key() -> (
    None
):
    fixture = renderer.FIXTURE_CONTRACT[0]
    _, first_caption, first_row = renderer._render(
        fixture, "confirmation", 0, CONFIRMATION_KEY
    )
    _, second_caption, second_row = renderer._render(
        fixture,
        "confirmation",
        0,
        b"independent-confirm-key-material-0003",
    )
    assert first_row["row_id"] != second_row["row_id"]
    assert first_row["relative_image_path"] != second_row["relative_image_path"]
    assert first_row["relative_caption_path"] != second_row["relative_caption_path"]
    assert first_caption != second_caption
    assert first_row["caption_sha256"] != second_row["caption_sha256"]
    assert (
        first_row["visible_glyph_transcript"] != second_row["visible_glyph_transcript"]
    )
    assert first_row["group_identity"] != second_row["group_identity"]
    predictable = f"{str(fixture['fixture_id']).lower()}-confirmation-c1-001"
    assert first_row["row_id"] != predictable
    assert second_row["row_id"] != predictable


def test_create_only_disjoint_outputs_and_tamper_detection(
    built, tmp_path: Path
) -> None:
    public, custodian, _ = built
    with pytest.raises(FileExistsError, match="refusing to replace"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(public.parent,),
            **RIGHTS,
        )
    with pytest.raises(renderer.FixtureError, match="custodian output must be outside"):
        renderer.build_candidate(
            public_output=tmp_path / "nested" / "public",
            custodian_output=tmp_path / "nested" / "custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(tmp_path / "nested",),
            **RIGHTS,
        )
    copied_public = tmp_path / "copied-boundary" / "copied-public"
    copied_custodian = tmp_path / "copied-custodian"
    copied_public.parent.mkdir()
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, copied_custodian)
    manifest = _json(copied_public / "DISCOVERY-MANIFEST.json")
    row = next(iter(manifest["families"].values()))["rows"][0]
    target = copied_public / "discovery" / row["relative_image_path"]
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(renderer.FixtureError, match="root inventory mismatch"):
        renderer.verify_candidate(
            copied_public,
            copied_custodian,
            public_boundary_roots=(copied_public.parent,),
        )

    clean_public = tmp_path / "extra-boundary" / "extra-public"
    clean_custodian = tmp_path / "extra-custodian"
    clean_public.parent.mkdir()
    shutil.copytree(public, clean_public)
    shutil.copytree(custodian, clean_custodian)
    (clean_public / "unexpected-private-review.json").write_text("{}\n")
    with pytest.raises(renderer.FixtureError, match="root inventory mismatch"):
        renderer.verify_candidate(
            clean_public,
            clean_custodian,
            public_boundary_roots=(clean_public.parent,),
        )


def test_custody_rejects_leaf_only_public_boundary_and_evidence_siblings(
    tmp_path: Path,
) -> None:
    leaf_public = tmp_path / "leaf-public"
    with pytest.raises(renderer.FixtureError, match="strictly inside"):
        renderer.build_candidate(
            public_output=leaf_public,
            custodian_output=tmp_path / "leaf-private",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(leaf_public,),
            **RIGHTS,
        )
    assert not leaf_public.exists()

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(renderer.FixtureError, match="custodian output must be outside"):
        renderer.build_candidate(
            public_output=evidence / "candidate",
            custodian_output=evidence / "private-confirmation",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(evidence,),
            **RIGHTS,
        )
    assert not (evidence / "candidate").exists()
    assert not (evidence / "private-confirmation").exists()


def test_custody_rejects_executable_repo_and_every_registered_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "upload"
    boundary.mkdir()
    with pytest.raises(renderer.FixtureError, match="custodian output must be outside"):
        renderer.build_candidate(
            public_output=boundary / "candidate-a",
            custodian_output=renderer.EXECUTABLE_REPOSITORY_ROOT / "private-a",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )

    second_worktree = tmp_path / "missing-prunable-second-worktree"
    monkeypatch.setattr(
        renderer,
        "_registered_worktree_roots",
        lambda: (renderer.EXECUTABLE_REPOSITORY_ROOT, second_worktree),
    )
    with pytest.raises(renderer.FixtureError, match="custodian output must be outside"):
        renderer.build_candidate(
            public_output=boundary / "candidate-b",
            custodian_output=second_worktree / "private-b",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert not (boundary / "candidate-a").exists()
    assert not (boundary / "candidate-b").exists()


def test_path_overlap_uses_filesystem_identity_for_case_aliases(tmp_path: Path) -> None:
    root = tmp_path / "CaseSensitiveProbe"
    root.mkdir()
    alias = root.with_name(root.name.swapcase())
    try:
        same = alias.samefile(root)
    except FileNotFoundError:
        pytest.skip("filesystem is case-sensitive")
    if not same:
        pytest.skip("filesystem is case-sensitive")
    assert renderer._paths_equivalent(alias, root)
    assert renderer._paths_overlap(alias / "private", root)
    assert renderer._path_is_descendant(alias / "private", root, strict=True)


def test_path_overlap_uses_terminal_identity_across_apfs_firmlink() -> None:
    canonical = renderer.EXECUTABLE_REPOSITORY_ROOT
    alias = Path("/System/Volumes/Data") / canonical.relative_to(canonical.anchor)
    try:
        same = alias.samefile(canonical)
    except FileNotFoundError:
        pytest.skip("APFS /System/Volumes/Data firmlink is unavailable")
    if not same:
        pytest.skip("alternate APFS path is not a filesystem alias")
    assert renderer._paths_equivalent(alias, canonical)
    assert renderer._paths_overlap(alias / "private", canonical)
    assert renderer._path_is_descendant(alias / "private", canonical, strict=True)


def test_path_identity_rejects_fifo_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = tmp_path / "custody-fifo"
    os.mkfifo(fifo)
    original_open = renderer.os.open
    observed_nonblock = False

    def require_nonblock(path, flags, *args, **kwargs):
        nonlocal observed_nonblock
        if path == fifo.name and kwargs.get("dir_fd") is not None:
            assert flags & os.O_NONBLOCK
            observed_nonblock = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(renderer.os, "open", require_nonblock)
    with pytest.raises(renderer.FixtureError, match="not a regular file or directory"):
        renderer._path_identity_plan(fifo)
    assert observed_nonblock is True


def test_new_root_cleanup_never_deletes_replacement_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "target"
    moved = parent / "created-by-function"
    original_open_chain = renderer._open_directory_chain_no_symlinks
    calls = 0

    def replace_before_fresh_open(path, label):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.rename(moved)
            target.mkdir()
        return original_open_chain(path, label)

    monkeypatch.setattr(
        renderer, "_open_directory_chain_no_symlinks", replace_before_fresh_open
    )
    with pytest.raises(renderer.FixtureError, match="absolute path changed"):
        renderer._ensure_new_root(target, "private root")
    assert target.is_dir()
    assert moved.is_dir()


def test_custodian_creation_rejects_parent_swap_to_public_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "public-boundary"
    boundary.mkdir()
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    abandoned_parent = tmp_path / "private-parent-abandoned"
    original_mkdir = renderer.os.mkdir
    swapped = False

    def swap_parent_before_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "custodian" and dir_fd is not None and not swapped:
            private_parent.rename(abandoned_parent)
            private_parent.symlink_to(boundary, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return original_mkdir(path, mode)
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(renderer.os, "mkdir", swap_parent_before_mkdir)
    with pytest.raises(renderer.FixtureError, match="cannot safely open custodian"):
        renderer.build_candidate(
            public_output=boundary / "candidate",
            custodian_output=private_parent / "custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert swapped is True
    assert not (boundary / "custodian").exists()
    # Exceptional cleanup is deliberately nondestructive: create-only debris
    # remains manifest-free and therefore cannot validate as a candidate.
    assert (abandoned_parent / "custodian").is_dir()
    assert not any((abandoned_parent / "custodian").iterdir())
    assert (boundary / "candidate").is_dir()
    assert not any((boundary / "candidate").iterdir())


def test_generation_rejects_custodian_move_into_public_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "public-boundary"
    boundary.mkdir()
    public = boundary / "candidate"
    custodian = tmp_path / "custodian"
    relocated = boundary / "relocated-private"
    tiny = dict(renderer.FIXTURE_CONTRACT[0])
    tiny.update(
        {
            "discovery_packs": [
                {"pack": "D1", "training_count": 1, "evaluation_count": 1}
            ],
            "confirmation_packs": [
                {"pack": "C1", "training_count": 1, "evaluation_count": 1}
            ],
            "discovery_count": 2,
            "confirmation_count": 2,
        }
    )
    monkeypatch.setattr(renderer, "FIXTURE_CONTRACT", (tiny,))
    original_validate = renderer._CustodySession.validate_private_publish
    moved = False

    def move_then_validate(self, descriptor, metadata):
        nonlocal moved
        if not moved:
            self.custodian.path.rename(relocated)
            moved = True
        return original_validate(self, descriptor, metadata)

    monkeypatch.setattr(
        renderer._CustodySession,
        "validate_private_publish",
        move_then_validate,
    )
    with pytest.raises(renderer.FixtureError, match="custodian output"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert moved is True
    assert not (public / "CANDIDATE-MANIFEST.json").exists()
    assert all(
        path.stat().st_size == 0 for path in relocated.rglob("*") if path.is_file()
    )


def test_generation_rejects_worktree_registered_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "anchor.txt").write_text("anchor\n", encoding="ascii")
    subprocess.run(["git", "add", "anchor.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SN56 Test",
            "-c",
            "user.email=sn56@example.invalid",
            "commit",
            "-qm",
            "anchor",
        ],
        cwd=repository,
        check=True,
    )
    boundary = tmp_path / "public-boundary"
    boundary.mkdir()
    worktree = tmp_path / "late-worktree"
    custodian = worktree / "private"
    original_create = renderer._BoundDirectory.create.__func__
    registered = False

    def register_then_create(cls, path, label):
        nonlocal registered
        if label == "custodian output" and not registered:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
            )
            registered = True
        return original_create(cls, path, label)

    monkeypatch.setattr(renderer, "EXECUTABLE_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        renderer._BoundDirectory,
        "create",
        classmethod(register_then_create),
    )
    with pytest.raises(renderer.FixtureError, match="custodian output must be outside"):
        renderer.build_candidate(
            public_output=boundary / "candidate",
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert registered is True
    assert not (boundary / "candidate" / "CANDIDATE-MANIFEST.json").exists()


def test_generation_completion_rechecks_custodian_after_inner_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "completion-public-boundary"
    boundary.mkdir()
    public = boundary / "candidate"
    custodian = tmp_path / "completion-custodian"
    relocated = boundary / "late-relocated-private"
    tiny = dict(renderer.FIXTURE_CONTRACT[0])
    tiny.update(
        {
            "discovery_packs": [
                {"pack": "D1", "training_count": 1, "evaluation_count": 1}
            ],
            "confirmation_packs": [
                {"pack": "C1", "training_count": 1, "evaluation_count": 1}
            ],
            "discovery_count": 2,
            "confirmation_count": 2,
        }
    )
    monkeypatch.setattr(renderer, "FIXTURE_CONTRACT", (tiny,))
    original_build = renderer._build_candidate_in_session
    moved = False

    def build_then_move(*args, **kwargs):
        nonlocal moved
        result = original_build(*args, **kwargs)
        custodian.rename(relocated)
        moved = True
        return result

    monkeypatch.setattr(renderer, "_build_candidate_in_session", build_then_move)
    with pytest.raises(renderer.FixtureError, match="custodian output"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert moved is True
    assert all(
        path.stat().st_size == 0 for path in relocated.rglob("*") if path.is_file()
    )


def test_generation_completion_rejects_public_hard_link_to_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "hardlink-public-boundary"
    boundary.mkdir()
    public = boundary / "candidate"
    custodian = tmp_path / "hardlink-custodian"
    public_link = boundary / "leaked-private-row"
    tiny = dict(renderer.FIXTURE_CONTRACT[0])
    tiny.update(
        {
            "discovery_packs": [
                {"pack": "D1", "training_count": 1, "evaluation_count": 1}
            ],
            "confirmation_packs": [
                {"pack": "C1", "training_count": 1, "evaluation_count": 1}
            ],
            "discovery_count": 2,
            "confirmation_count": 2,
        }
    )
    monkeypatch.setattr(renderer, "FIXTURE_CONTRACT", (tiny,))
    original_build = renderer._build_candidate_in_session

    def build_then_link(*args, **kwargs):
        result = original_build(*args, **kwargs)
        private_file = next(
            path for path in sorted(custodian.rglob("*")) if path.is_file()
        )
        os.link(private_file, public_link)
        return result

    monkeypatch.setattr(renderer, "_build_candidate_in_session", build_then_link)
    with pytest.raises(renderer.FixtureError, match="moved or linked before completion"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert public_link.read_bytes() == b""


def test_generation_completion_rejects_private_subtree_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "subtree-public-boundary"
    boundary.mkdir()
    public = boundary / "candidate"
    custodian = tmp_path / "subtree-custodian"
    relocated = boundary / "relocated-confirmation-subtree"
    tiny = dict(renderer.FIXTURE_CONTRACT[0])
    tiny.update(
        {
            "discovery_packs": [
                {"pack": "D1", "training_count": 1, "evaluation_count": 1}
            ],
            "confirmation_packs": [
                {"pack": "C1", "training_count": 1, "evaluation_count": 1}
            ],
            "discovery_count": 2,
            "confirmation_count": 2,
        }
    )
    monkeypatch.setattr(renderer, "FIXTURE_CONTRACT", (tiny,))
    original_build = renderer._build_candidate_in_session

    def build_then_move_subtree(*args, **kwargs):
        result = original_build(*args, **kwargs)
        (custodian / "confirmation").rename(relocated)
        return result

    monkeypatch.setattr(
        renderer, "_build_candidate_in_session", build_then_move_subtree
    )
    with pytest.raises(renderer.FixtureError, match="subtree moved before completion"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert all(
        path.stat().st_size == 0 for path in relocated.rglob("*") if path.is_file()
    )


def test_generation_completion_rejects_private_file_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "file-public-boundary"
    boundary.mkdir()
    public = boundary / "candidate"
    custodian = tmp_path / "file-custodian"
    relocated = boundary / "relocated-private-file"
    tiny = dict(renderer.FIXTURE_CONTRACT[0])
    tiny.update(
        {
            "discovery_packs": [
                {"pack": "D1", "training_count": 1, "evaluation_count": 1}
            ],
            "confirmation_packs": [
                {"pack": "C1", "training_count": 1, "evaluation_count": 1}
            ],
            "discovery_count": 2,
            "confirmation_count": 2,
        }
    )
    monkeypatch.setattr(renderer, "FIXTURE_CONTRACT", (tiny,))
    original_build = renderer._build_candidate_in_session

    def build_then_move_file(*args, **kwargs):
        result = original_build(*args, **kwargs)
        private_file = next(
            path for path in sorted(custodian.rglob("*")) if path.is_file()
        )
        private_file.rename(relocated)
        return result

    monkeypatch.setattr(renderer, "_build_candidate_in_session", build_then_move_file)
    with pytest.raises(renderer.FixtureError, match="file moved before completion"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(boundary,),
            **RIGHTS,
        )
    assert relocated.read_bytes() == b""


def test_custody_rejects_symlink_ancestors_before_creation(tmp_path: Path) -> None:
    real_boundary = tmp_path / "real-public-boundary"
    real_boundary.mkdir()
    linked_boundary = tmp_path / "linked-public-boundary"
    linked_boundary.symlink_to(real_boundary, target_is_directory=True)
    with pytest.raises(renderer.FixtureError, match="symlink component"):
        renderer.build_candidate(
            public_output=linked_boundary / "candidate",
            custodian_output=tmp_path / "private-a",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(linked_boundary,),
            **RIGHTS,
        )

    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    linked_private = tmp_path / "linked-private"
    linked_private.symlink_to(private_parent, target_is_directory=True)
    with pytest.raises(renderer.FixtureError, match="symlink component"):
        renderer.build_candidate(
            public_output=real_boundary / "candidate",
            custodian_output=linked_private / "custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(real_boundary,),
            **RIGHTS,
        )
    assert not (real_boundary / "candidate").exists()


def test_relocated_private_tree_inside_public_boundary_fails_verify_and_admission(
    built, tmp_path: Path
) -> None:
    public, custodian, _ = built
    evidence = tmp_path / "relocated-evidence"
    evidence.mkdir()
    copied_public = evidence / "candidate"
    relocated_custodian = evidence / "private"
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, relocated_custodian)
    with pytest.raises(renderer.FixtureError, match="custodian output must be outside"):
        renderer.verify_candidate(
            copied_public,
            relocated_custodian,
            public_boundary_roots=(evidence,),
        )


def test_fixed_git_worktree_inventory_is_config_isolated_and_malformed_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, *, cwd, env, check, capture_output):
        observed.update(
            argv=argv, cwd=cwd, env=env, check=check, capture=capture_output
        )
        output = (
            f"worktree {renderer.EXECUTABLE_REPOSITORY_ROOT}\0" f"HEAD {'a' * 40}\0\0"
        ).encode()
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    assert renderer._registered_worktree_roots() == (
        renderer.EXECUTABLE_REPOSITORY_ROOT,
    )
    assert observed["argv"][0] == "/usr/bin/git"
    assert observed["argv"][-4:] == ["worktree", "list", "--porcelain", "-z"]
    assert observed["env"] == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-hke-custody",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }

    with pytest.raises(renderer.FixtureError, match="malformed"):
        renderer._parse_worktree_roots(b"HEAD " + b"b" * 40 + b"\0\0")


def test_source_and_records_exclude_forbidden_input_surfaces(built) -> None:
    _, _, result = built
    assert result["forbidden_inventories"] == renderer.FORBIDDEN_INVENTORIES
    assert all(value == [] for value in result["forbidden_inventories"].values())
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "import random",
        "import time",
        "import urllib",
        "import requests",
        "ImageFont",
        "os.environ",
        "os.getenv",
    ):
        assert forbidden not in source
    assert result["generator"]["network_access"] == "no-network-code-path"
    assert result["generator"]["ambient_inputs"] == []
    assert any(
        item.startswith("Pillow==") for item in result["generator"]["dependencies"]
    )


def test_owner_and_use_record_are_mandatory_before_candidate_creation(
    tmp_path: Path,
) -> None:
    invalid = dict(RIGHTS)
    invalid["rights_owner"] = "TBD"
    with pytest.raises(renderer.FixtureError, match="rights owner"):
        renderer.build_candidate(
            public_output=tmp_path / "rights-boundary" / "public",
            custodian_output=tmp_path / "custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            public_boundary_roots=(tmp_path / "rights-boundary",),
            **invalid,
        )


def test_declarative_contract_matches_the_renderer_and_keeps_authority_closed() -> None:
    contract = json.loads(
        MODULE_PATH.with_name("hke_fixture_contract.json").read_text(encoding="utf-8")
    )
    expected = {
        item["family"]: {
            "discovery": {
                record["pack"]: {
                    "training": record["training_count"],
                    "evaluation": record["evaluation_count"],
                }
                for record in item["discovery_packs"]
            },
            "confirmation": {
                record["pack"]: {
                    "training": record["training_count"],
                    "evaluation": record["evaluation_count"],
                }
                for record in item["confirmation_packs"]
            },
        }
        for item in renderer.FIXTURE_CONTRACT
    }
    actual = {
        family: {
            "discovery": value["discovery_packs"],
            "confirmation": value["confirmation_packs"],
        }
        for family, value in contract["families"].items()
    }
    assert actual == expected
    assert contract["authorization"] == {
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "agent_review_is_not_human_review": True,
        "human_review_evidence_class": "operator_attested",
        "owner_ratification_required_for_gpu": True,
    }
    assert contract["schema"] == renderer.SCHEMA == 3
