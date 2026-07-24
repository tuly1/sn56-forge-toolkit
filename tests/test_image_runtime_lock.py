from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "ops/docker/image-runtime-lock.txt"
CONSTRAINTS = ROOT / "ops/docker/image-runtime-phase1-constraints.txt"
VERIFIER = ROOT / "ops/docker/verify_image_runtime.py"
TOOLKIT_DOCKERFILE = (
    ROOT / "ops/docker/standalone-image-toolkit-trainer.dockerfile"
)
LEGACY_FLUX_DOCKERFILE = ROOT / "ops/docker/standalone-image-trainer.dockerfile"

SPEC = importlib.util.spec_from_file_location("verify_image_runtime", VERIFIER)
assert SPEC is not None and SPEC.loader is not None
verify_image_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_image_runtime)


def _result(returncode: int, stdout: str, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def _runner(*results):
    calls = []
    queue = list(results)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return queue.pop(0)

    run.calls = calls
    return run


def test_certified_lock_is_exact_evidence_inventory():
    lines = LOCK.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 185
    assert (
        verify_image_runtime.GOLDEN_INVENTORY_SHA256
        == "9c4c15130508c547c67d891f559ca1a513cd62bd5a4b695eb25ceafccd0b850b"
    )
    assert (
        "easy_dwpose @ git+https://github.com/jaretburkett/easy_dwpose.git@"
        "028aa1449f9e07bdeef7f84ed0ce7a2660e72239"
    ) in lines
    assert "huggingface_hub==1.10.1" in lines


def test_phase1_constraints_are_the_exact_mechanical_derivation():
    inventory = LOCK.read_text(encoding="utf-8")
    constraints = CONSTRAINTS.read_text(encoding="utf-8")

    assert constraints == verify_image_runtime.derive_phase1_constraints(inventory)
    assert len(constraints.splitlines()) == 162
    assert (
        verify_image_runtime.PHASE1_CONSTRAINTS_SHA256
        == "864ed2d3c45f86464b189e3f1685e0578eae2af9ecf49e6bb63cadf3a85986ac"
    )

    excluded = [
        line
        for line in inventory.splitlines()
        if verify_image_runtime.phase1_excludes(line)
    ]
    excluded_names = {
        verify_image_runtime.distribution_name(line) for line in excluded
    }
    prefixed_names = {
        name
        for name in excluded_names
        if name.startswith("nvidia-") or name.startswith("cuda-")
    }
    assert len(excluded) == 23
    assert excluded_names - prefixed_names == set(
        verify_image_runtime.PHASE1_EXCLUDED_NAMES
    )
    assert not any(
        verify_image_runtime.phase1_excludes(line)
        for line in constraints.splitlines()
    )


def test_toolkit_image_applies_the_certified_two_phase_lock():
    contents = TOOLKIT_DOCKERFILE.read_text(encoding="utf-8")

    assert contents.count("Phase 1") == 1
    assert contents.count("Phase 2") == 1
    assert "--no-deps" in contents
    assert "--requirement /opt/sn56/image-runtime-lock.txt" in contents
    assert "python3 /opt/sn56/verify-image-runtime.py" in contents
    assert (
        contents.count(
            "--constraint /opt/sn56/image-runtime-phase1-constraints.txt"
        )
        == 2
    )
    assert "--files-only" in contents


def test_legacy_flux_image_carries_two_pinned_isolated_runtimes():
    contents = LEGACY_FLUX_DOCKERFILE.read_text(encoding="utf-8")

    assert contents != TOOLKIT_DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "FROM diagonalge/ai-toolkit:latest@sha256:"
        "c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8 "
        "AS aitoolkit-runtime"
    ) in contents
    assert (
        "FROM diagonalge/kohya_latest:latest@sha256:"
        "d34dd5750e1018455e111f63c03bb2a4e16204607e00ba5af870dd7c71beb84e"
    ) in contents
    assert "FORGE_FLUX_BACKEND=kohya" in contents
    assert "AI_TOOLKIT_DIR=/app/ai-toolkit" in contents
    assert "PYTHONPATH=/opt/sn56/ai-toolkit-python" in contents
    assert "PYTHONNOUSERSITE=1" in contents
    assert (
        "FORGE_KOHYA_PYTHONPATH=/home/.local/lib/python3.10/site-packages"
        in contents
    )
    assert "FORGE_KOHYA_LD_PRELOAD=libtcmalloc.so" in contents
    assert "FORGE_KOHYA_PROTOBUF_IMPLEMENTATION=python" in contents
    assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb" in contents
    assert (
        "COPY --from=aitoolkit-runtime /app/ai-toolkit/ /app/ai-toolkit/"
        in contents
    )
    assert (
        "COPY --from=aitoolkit-runtime "
        "/usr/local/lib/python3.10/dist-packages/ "
        "/opt/sn56/ai-toolkit-python/"
    ) in contents
    for system_distribution, target_name in (
        ("pip", "pip"),
        ("pip-22.0.2.dist-info", "pip-22.0.2.dist-info"),
        ("wheel", "wheel"),
        ("wheel-0.37.1.egg-info", "wheel-0.37.1.egg-info"),
    ):
        assert (
            "COPY --from=aitoolkit-runtime "
            f"/usr/lib/python3/dist-packages/{system_distribution}/ "
            f"/opt/sn56/ai-toolkit-python/{target_name}/"
        ) in contents
    assert contents.count("99be3d96a2468d3a5228a4eb05ba67e63c586b4e") == 3
    assert "--requirement /opt/sn56/image-runtime-lock.txt" in contents
    assert "python3 /opt/sn56/verify-image-runtime.py" in contents
    assert "assert torch.__version__ == '2.6.0+cu124'" in contents
    assert "assert torch.version.cuda == '12.4'" in contents
    assert "startswith('/opt/sn56/ai-toolkit-python/')" in contents
    assert "SD_SCRIPTS_DIR=/app/sd-scripts" in contents
    assert "flux_train_network.py" in contents
    assert "python3 -m forge.verify_flux_kohya_runtime" in contents
    assert "python3 -m forge.flux_kohya_tokenizers stage" in contents
    assert "python3 -m forge.flux_kohya_tokenizers verify" in contents
    assert "32bd64288804d66eefd0ccbe215aa642df71cc41" in (
        ROOT / "forge/flux_kohya_tokenizers.py"
    ).read_text(encoding="utf-8")
    assert "3db67ab1af984cf10548a73467f0e5bca2aaaeb2" in (
        ROOT / "forge/flux_kohya_tokenizers.py"
    ).read_text(encoding="utf-8")
    assert (
        'ENTRYPOINT ["dumb-init", "--", "python3", "-m", "forge.cli"]'
        in contents
    )
    assert "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38" in contents
    assert "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd" in contents
    assert "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635" in contents
    assert "assert torch.__version__ == '2.1.2+cu121'" in contents
    assert "assert torch.version.cuda == '12.1'" in contents
    assert "startswith('/home/.local/lib/python3.10/site-packages/')" in contents


def test_image_sources_are_pinned_to_certified_identities():
    dockerfile = TOOLKIT_DOCKERFILE.read_text(encoding="utf-8")
    inventory = LOCK.read_text(encoding="utf-8")

    assert (
        "FROM diagonalge/ai-toolkit:latest@sha256:"
        "c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8"
    ) in dockerfile
    assert dockerfile.count("99be3d96a2468d3a5228a4eb05ba67e63c586b4e") == 2
    assert (
        "diffusers @ git+https://github.com/huggingface/diffusers.git@"
        "dc8d9032171c83741fd37ed2b12bc9d8274464f3"
    ) in inventory
    assert (
        "easy_dwpose @ git+https://github.com/jaretburkett/easy_dwpose.git@"
        "028aa1449f9e07bdeef7f84ed0ce7a2660e72239"
    ) in inventory


def test_runtime_lock_wording_is_limited_to_metadata_inventory():
    implementation = "\n".join(
        [
            TOOLKIT_DOCKERFILE.read_text(encoding="utf-8"),
            VERIFIER.read_text(encoding="utf-8"),
        ]
    ).lower()

    assert "version/vcs metadata inventory" in implementation
    assert "does not attest downloaded wheel bytes" in implementation
    assert "hermetic" not in implementation
    assert "byte-identical" not in implementation
    assert "overlays every" not in implementation


def test_verifier_accepts_only_the_certified_runtime_and_conflict():
    expected_freeze = LOCK.read_text(encoding="utf-8")
    allowed_conflict = verify_image_runtime.ALLOWED_PIP_CHECK_LINES[0]
    run = _runner(_result(0, expected_freeze), _result(1, allowed_conflict + "\n"))

    result = verify_image_runtime.verify_runtime(
        LOCK, CONSTRAINTS, runner=run, python_executable="/runtime/python"
    )

    assert result == {
        "result": "PASS",
        "verification_scope": "pip-freeze-version-vcs-metadata",
        "inventory_sha256": verify_image_runtime.GOLDEN_INVENTORY_SHA256,
        "inventory_entry_count": 185,
        "phase1_constraints_sha256": verify_image_runtime.PHASE1_CONSTRAINTS_SHA256,
        "phase1_constraint_count": 162,
        "allowed_pip_check_conflicts": [allowed_conflict],
    }
    assert [call[0] for call in run.calls] == [
        ["/runtime/python", "-m", "pip", "freeze", "--all"],
        ["/runtime/python", "-m", "pip", "check"],
    ]


def test_verifier_rejects_a_runtime_freeze_mismatch():
    run = _runner(_result(0, LOCK.read_text(encoding="utf-8") + "drift==1\n"))

    with pytest.raises(
        verify_image_runtime.VerificationError, match="metadata inventory mismatch"
    ):
        verify_image_runtime.verify_runtime(LOCK, CONSTRAINTS, runner=run)


def test_verifier_rejects_pip_diagnostics_on_stderr():
    run = _runner(_result(0, LOCK.read_text(encoding="utf-8"), "invalid metadata\n"))

    with pytest.raises(verify_image_runtime.VerificationError, match="freeze emitted stderr"):
        verify_image_runtime.verify_runtime(LOCK, CONSTRAINTS, runner=run)


@pytest.mark.parametrize(
    ("returncode", "output"),
    [
        (0, ""),
        (
            1,
            verify_image_runtime.ALLOWED_PIP_CHECK_LINES[0]
            + "\nanother 1 has requirement missing>=2, but you have missing 1.\n",
        ),
        (1, ""),
    ],
)
def test_verifier_rejects_any_other_pip_check_state(returncode, output):
    run = _runner(
        _result(0, LOCK.read_text(encoding="utf-8")),
        _result(returncode, output),
    )

    with pytest.raises(verify_image_runtime.VerificationError, match="pip check"):
        verify_image_runtime.verify_runtime(LOCK, CONSTRAINTS, runner=run)


def test_verifier_rejects_an_edited_golden_lock(tmp_path):
    edited_lock = tmp_path / "image-runtime-lock.txt"
    edited_lock.write_text(LOCK.read_text(encoding="utf-8") + "drift==1\n")

    with pytest.raises(verify_image_runtime.VerificationError, match="digest mismatch"):
        verify_image_runtime.verify_runtime(edited_lock, CONSTRAINTS, runner=_runner())


def test_verifier_rejects_an_edited_phase1_constraints(tmp_path):
    edited_constraints = tmp_path / "image-runtime-phase1-constraints.txt"
    edited_constraints.write_text(
        CONSTRAINTS.read_text(encoding="utf-8") + "drift==1\n"
    )

    with pytest.raises(verify_image_runtime.VerificationError, match="digest mismatch"):
        verify_image_runtime.verify_metadata_files(LOCK, edited_constraints)


def test_allowed_pip_check_conflict_is_exact_and_singular():
    assert verify_image_runtime.ALLOWED_PIP_CHECK_LINES == (
        "easy-dwpose 1.0.3 has requirement huggingface_hub<1.0,>=0.26, "
        "but you have huggingface-hub 1.10.1.",
    )
