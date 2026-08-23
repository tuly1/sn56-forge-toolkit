"""CPU-only, network-free tests for the manifest-bound Sunday/Monday probe."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "sn56-preentry-probe-v2.sh"
WRAPPER = ROOT / "scripts" / "sn56-monday-probe.sh"
MANIFEST_PATH = ROOT / "release" / "week9-release-manifest.json"
DOCKER_POLICY_PATH = ROOT / "release" / "week9-docker-policy.json"
READINESS_PATH = ROOT / "release" / "week9-release-readiness.json"
SELECTED_MANIFEST_PATH = ROOT / "tests" / "data" / "week9-release-selected-hold.json"
MANIFEST = json.loads(SELECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
TARGET = MANIFEST["target"]["commit"]
ROLLBACK = MANIFEST["rollback"]["commit"]
TEXT_PIN = "8f11684e30a556b305dec9dd8eec9794bdae8cde"
NOW = "2026-08-21T09:00:00Z"
PRODUCTION = MANIFEST["production"]
SERVICE_PYTHON = str(
    Path(shlex.split(PRODUCTION["service_exec_start"])[0]).with_name("python")
)
SERVICE_PYTHON_REALPATH = "/opt/python/cpython-3.12/bin/python3.12"
ENDPOINT_URL = (
    f"http://{PRODUCTION['endpoint_host']}:{PRODUCTION['endpoint_port']}"
    f"{PRODUCTION['endpoint_route']}"
)
TRACKED_BASELINE = ROOT / "scripts" / "sn56-upstream-baseline.env"


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _manifest() -> dict:
    return json.loads(SELECTED_MANIFEST_PATH.read_text(encoding="utf-8"))


def _fiber_v27_body() -> str:
    detail = [
        {
            "type": "missing",
            "loc": ["header", "validator-hotkey"],
            "msg": "Field required",
        },
        {
            "type": "missing",
            "loc": ["header", "validator-hotkey"],
            "msg": "Field required",
        },
        {"type": "missing", "loc": ["header", "signature"]},
        {"type": "missing", "loc": ["header", "miner-hotkey"]},
        {"type": "missing", "loc": ["header", "nonce"]},
    ]
    return json.dumps(
        {"detail": detail}
    )


def _pin_evidence() -> dict:
    return {
        "repo_keys": ["IMAGE", "TEXT"],
        "source_image_pin": TARGET,
        "source_image_repo": "https://github.com/tuly1/sn56-forge-toolkit",
        "source_text_pin": TEXT_PIN,
        "source_target_count": 1,
        "source_rollback_count": 0,
        "source_text_count": 1,
        "pyc_target_count": 1,
        "pyc_rollback_count": 0,
        "pyc_text_count": 1,
        "pid_before": "4242",
        "pid_after": "4242",
        "service_state_before": "active",
        "service_state_after": "active",
        "service_user": PRODUCTION["service_user"],
        "service_working_directory": PRODUCTION["service_working_directory"],
        "service_exec_argv": PRODUCTION["service_exec_start"],
        "process_cwd": PRODUCTION["service_working_directory"],
        "process_cmdline": [
            SERVICE_PYTHON,
            *shlex.split(PRODUCTION["service_exec_start"]),
        ],
        "reviewed_python": SERVICE_PYTHON,
        "reviewed_python_realpath": SERVICE_PYTHON_REALPATH,
        "process_exe": SERVICE_PYTHON_REALPATH,
        "process_exe_realpath": SERVICE_PYTHON_REALPATH,
        "process_pythonpath": None,
        "resolved_asgi_module": PRODUCTION["service_asgi_module"],
        "resolved_training_repo_module": PRODUCTION["endpoint_source"],
        "listener_pids": [4242],
        "listener_bound": True,
        "process_start_epoch": 1_787_000_100.0,
        "source_mtime_epoch": 1_787_000_000.0,
        "pyc_mtime_epoch": 1_787_000_101.0,
        "loopback_url": (
            f"http://127.0.0.1:{PRODUCTION['endpoint_port']}"
            f"{PRODUCTION['endpoint_route']}"
        ),
        "loopback_code": 422,
        "loopback_body": _fiber_v27_body(),
        "loopback_error": None,
    }


def _green_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(), indent=2) + "\n", encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()

    _write(
        fixtures / "upstream_constants.body",
        "TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK = 0\n"
        "TOURNAMENT_SCHEDULE_IMAGE_HOUR = 13\n",
    )
    _write(
        fixtures / "local_sched.out",
        "TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK=0\n"
        "TOURNAMENT_SCHEDULE_IMAGE_HOUR=13\n",
    )
    _write(
        fixtures / "latest_details.body",
        json.dumps({"image": {"tournament_id": "tourn_previous", "status": "completed"}}),
    )
    _write(fixtures / "chain_uid.out", "UID=224\n")
    _write(fixtures / "fees.body", json.dumps({"image_tournament_fee_rao": 200_000_000}))
    _write(fixtures / "balance.body", json.dumps({"balance_rao": 200_000_000}))
    _write(fixtures / "endpoint.url", ENDPOINT_URL + "\n")
    _write(fixtures / "endpoint.code", "422\n")
    _write(fixtures / "endpoint.body", _fiber_v27_body())
    _write(fixtures / "endpoint_pin.out", json.dumps(_pin_evidence()))
    _write(fixtures / "miner_state.out", "active 0\n")
    _write(
        fixtures / "metagraph.out",
        json.dumps(
            {
                "__REALTIME_TIMESTAMP": str(int((time.time() - 30) * 1_000_000)),
                "MESSAGE": "Successfully synced 256 nodes!",
            }
        )
        + "\n",
    )
    upstream = "f" * 40
    _write(fixtures / "upstream_head.body", json.dumps({"sha": upstream}))
    _write(fixtures / "disk.out", "385\n")
    _write(fixtures / "mem.out", "61000\n")
    _write(
        fixtures / "watcher_units.out",
        "sn56-week9-watcher.service\tactive\t"
        "/opt/sn56-watcher/watcher.py run "
        "--tournament-date 20260824 "
        "--start-at 2026-08-24T12:30:00Z "
        "--entry-capture-at 2026-08-24T14:00:00Z "
        "--hard-stop-at 2026-09-01T15:00:00Z\n",
    )
    baseline = tmp_path / "baseline.env"
    _write(
        baseline,
        f'UPSTREAM_REVIEWED_SHA="{upstream}"\nUPSTREAM_REVIEWED_AT="2026-08-20"\n',
    )
    return manifest, fixtures, baseline


def _run_probe(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    manifest, fixtures, baseline = _green_case(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            "--manifest",
            str(manifest),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            NOW,
            "--json",
            str(tmp_path / "result.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "SN56_BASELINE_ENV": str(baseline)},
        text=True,
        capture_output=True,
        check=False,
    )
    return result


def _state(output: str, check_id: str) -> str:
    match = re.search(rf"^(PASS|FAIL|WARN) +{re.escape(check_id)} +", output, re.MULTILINE)
    assert match, output
    return match.group(1)


def test_bd_hold_manifest_and_auth_stage_422_are_green(tmp_path: Path) -> None:
    result = _run_probe(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.reachable") == "PASS"
    assert _state(result.stdout, "endpoint.pin") == "PASS"
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["target_commit"] == TARGET
    assert receipt["fails"] == 0


def test_bare_probe_uses_the_tracked_upstream_baseline(tmp_path: Path) -> None:
    manifest, fixtures, _ = _green_case(tmp_path)
    match = re.search(
        r'^UPSTREAM_REVIEWED_SHA="([0-9a-f]{40})"$',
        TRACKED_BASELINE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "tracked probe baseline must contain one exact reviewed SHA"
    _write(fixtures / "upstream_head.body", json.dumps({"sha": match.group(1)}))
    env = os.environ.copy()
    env.pop("SN56_BASELINE_ENV", None)
    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            "--manifest",
            str(manifest),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            NOW,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _state(result.stdout, "upstream.baseline") == "PASS"


def test_last_weeks_deduction_cannot_hide_an_unfunded_next_entry(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    _write(
        fixtures / "balance.body",
        json.dumps(
            {
                "balance_rao": 0,
                "updated_at": "2026-08-17T13:05:00Z",
            }
        ),
    )
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _state(result.stdout, "entry.balance") == "FAIL"
    assert "OWNER MUST SEND" in result.stdout


def test_current_entry_window_deduction_is_not_a_false_balance_alarm(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    _write(
        fixtures / "balance.body",
        json.dumps(
            {
                "balance_rao": 0,
                "updated_at": "2026-08-24T13:05:00Z",
            }
        ),
    )
    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            "--manifest",
            str(manifest),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            "2026-08-24T14:00:00Z",
        ],
        cwd=ROOT,
        env={**os.environ, "SN56_BASELINE_ENV": str(baseline)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _state(result.stdout, "entry.balance") == "PASS"
    assert "fee already DEDUCTED for this cycle" in result.stdout


def test_ready_manifest_uses_the_same_exact_probe_contract(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["release_state"] = "ready"
    manifest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.pin") == "PASS"


def test_unknown_release_state_is_rejected_before_io(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["release_state"] = "ship"
    manifest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 2
    assert "release_state must be exactly 'hold' or 'ready'" in result.stderr
    assert "endpoint.reachable" not in result.stdout


def test_live_probe_refuses_the_checked_in_hold_artifacts_before_io(tmp_path: Path) -> None:
    manifest, _, _ = _green_case(tmp_path)
    env = os.environ.copy()
    env.pop("SN56_BASELINE_ENV", None)
    result = subprocess.run(
        ["bash", str(PROBE), "--manifest", str(manifest)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "live probe requires release_state=ready" in result.stderr
    assert "endpoint.reachable" not in result.stdout


def test_duplicate_manifest_keys_are_rejected_before_io(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    raw = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        raw.replace(
            '  "schema_version": 1,',
            '  "schema_version": 1,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 2
    assert "duplicate JSON key" in result.stderr
    assert "endpoint.reachable" not in result.stdout


@pytest.mark.parametrize(
    "mutation",
    ["rollback", "repository", "docker"],
)
def test_probe_reuses_the_full_fixed_release_schema(
    tmp_path: Path, mutation: str
) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "rollback":
        doc["rollback"]["commit"] = "1" * 40
        doc["allowed_changes"]["base_commit"] = doc["rollback"]["commit"]
    elif mutation == "repository":
        doc["source"]["repository_url"] = "https://example.invalid/unreviewed.git"
    else:
        doc["dockerfiles"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 2
    assert "release artifact error" in result.stderr
    assert "endpoint.reachable" not in result.stdout


@pytest.mark.parametrize(
    ("args", "env_name", "message"),
    [
        (["--now", NOW], None, "--now is a mock-only test hook"),
        (["--skip-chain"], None, "--skip-chain is a mock-only test hook"),
        ([], "SN56_BASELINE_ENV", "SN56_BASELINE_ENV is a mock-only test hook"),
    ],
    ids=["backdated-clock", "skipped-registration", "baseline-source-override"],
)
def test_live_probe_rejects_mock_only_escape_hatches_before_io(
    tmp_path: Path, args: list[str], env_name: str | None, message: str
) -> None:
    manifest, _, baseline = _green_case(tmp_path)
    env = os.environ.copy()
    env.pop("SN56_BASELINE_ENV", None)
    if env_name:
        env[env_name] = str(baseline)
    result = subprocess.run(
        ["bash", str(PROBE), "--manifest", str(manifest), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "endpoint.reachable" not in result.stdout


@pytest.mark.parametrize(
    "changes",
    [
        {
            "source_image_pin": ROLLBACK,
            "source_target_count": 0,
            "source_rollback_count": 1,
            "pyc_target_count": 0,
            "pyc_rollback_count": 1,
        },
        {"pyc_target_count": 0, "pyc_rollback_count": 1},
        {"pid_after": "9001"},
        {"listener_bound": False},
        {"process_start_epoch": 1_786_999_000.0},
        {"source_image_repo": "https://github.com/example/wrong-repo"},
        {"repo_keys": ["IMAGE", "TEXT", "ENVIRONMENT"]},
        {"service_user": "root"},
        {"service_working_directory": "/tmp/alternate-god"},
        {"service_exec_argv": "/usr/bin/python /tmp/fake.py"},
        {"process_cwd": "/tmp/alternate-god"},
        {"process_cmdline": ["/usr/bin/python", "/tmp/fake.py"]},
        {
            "process_cmdline": [
                "/tmp/unreviewed-interpreter",
                *shlex.split(PRODUCTION["service_exec_start"]),
            ]
        },
        {"process_exe_realpath": "/tmp/unreviewed-interpreter"},
        {"process_pythonpath": "/tmp/shadow-modules"},
        {"resolved_asgi_module": "/tmp/miner/asgi.py"},
        {"resolved_training_repo_module": "/tmp/miner/endpoints/training_repo.py"},
        {"loopback_code": 400},
        {
            "loopback_body": json.dumps(
                {
                    "detail": [
                        {
                            "type": "enum",
                            "loc": ["path", "task_type"],
                        }
                    ]
                }
            )
        },
    ],
    ids=[
        "wrong-served-pin",
        "stale-running-pyc",
        "service-restarted-during-sample",
        "listener-not-owned-by-service",
        "process-predates-installed-source",
        "wrong-served-repository",
        "environment-route-present",
        "wrong-service-user",
        "wrong-unit-working-directory",
        "wrong-unit-exec-start",
        "wrong-process-cwd",
        "wrong-process-command-line",
        "unreviewed-interpreter-prefix",
        "wrong-process-executable",
        "pythonpath-import-redirect",
        "wrong-asgi-module-resolution",
        "wrong-route-module-resolution",
        "wrong-loopback-status",
        "loopback-enum-rejection",
    ],
)
def test_wrong_or_unbound_served_pin_fails(tmp_path: Path, changes: dict) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    evidence = _pin_evidence()
    evidence.update(changes)
    _write(fixtures / "endpoint_pin.out", json.dumps(evidence))
    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            "--manifest",
            str(manifest),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            NOW,
        ],
        cwd=ROOT,
        env={**os.environ, "SN56_BASELINE_ENV": str(baseline)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.pin") == "FAIL"


def test_enum_rejection_422_is_not_route_success(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    _write(
        fixtures / "endpoint.body",
        json.dumps(
            {
                "detail": [
                    {
                        "type": "enum",
                        "loc": ["path", "task_type"],
                        "msg": "Input should be 'text', 'image' or 'environment'",
                    },
                    {"type": "missing", "loc": ["header", "validator-hotkey"]},
                ]
            }
        ),
    )
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.reachable") == "FAIL"
    assert "path/enum validation" in result.stdout


def test_observed_duplicate_validator_hotkey_is_exact_route_success(
    tmp_path: Path,
) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    observed = _fiber_v27_body()
    _write(fixtures / "endpoint.body", observed)
    pin_evidence = _pin_evidence()
    pin_evidence["loopback_body"] = observed
    _write(fixtures / "endpoint_pin.out", json.dumps(pin_evidence))

    result = _run_existing_case(manifest, fixtures, baseline)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.reachable") == "PASS"
    assert _state(result.stdout, "endpoint.pin") == "PASS"
    assert "validator-hotkey': 2" in result.stdout


@pytest.mark.parametrize(
    "mutation",
    ["obsolete-four", "wrong-header-duplicate", "third-validator-hotkey"],
)
def test_only_observed_validator_hotkey_duplicate_is_tolerated(
    tmp_path: Path, mutation: str
) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    payload = json.loads(_fiber_v27_body())
    if mutation == "obsolete-four":
        payload["detail"].pop(0)
    elif mutation == "wrong-header-duplicate":
        payload["detail"].append(
            {"type": "missing", "loc": ["header", "signature"]}
        )
    else:
        payload["detail"].append(
            {"type": "missing", "loc": ["header", "validator-hotkey"]}
        )
    _write(fixtures / "endpoint.body", json.dumps(payload))

    result = _run_existing_case(manifest, fixtures, baseline)

    assert result.returncode == 1, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.reachable") == "FAIL"
    assert "missing-header multiset differs" in result.stdout


@pytest.mark.parametrize(
    ("code", "body"),
    [
        ("422", "not-json"),
        ("422", json.dumps({"detail": []})),
        (
            "422",
            json.dumps(
                {"detail": [{"type": "missing", "loc": ["header", "X-Unrelated"]}]}
            ),
        ),
        (
            "422",
            json.dumps(
                {
                    "detail": [
                        {
                            "type": "missing",
                            "loc": ["header", "validator-hotkey"],
                        }
                    ]
                }
            ),
        ),
        (
            "422",
            json.dumps(
                {
                    "detail": [
                        {
                            "type": "MISSING",
                            "loc": ["header", name],
                        }
                        for name in (
                            "validator-hotkey",
                            "signature",
                            "miner-hotkey",
                            "nonce",
                        )
                    ]
                }
            ),
        ),
        (
            "422",
            json.dumps(
                {
                    "detail": [
                        {
                            "type": "not_missing_but_fake",
                            "loc": ["header", name, "unexpected"],
                        }
                        for name in (
                            "validator-hotkey",
                            "signature",
                            "miner-hotkey",
                            "nonce",
                        )
                    ]
                }
            ),
        ),
        (
            "400",
            json.dumps(
                {
                    "detail": [
                        {"type": "missing", "loc": ["header", "validator-hotkey"]}
                    ]
                }
            ),
        ),
    ],
    ids=[
        "malformed-json",
        "missing-detail",
        "unrelated-header",
        "partial-fiber-header-contract",
        "forged-missing-types-and-locations",
        "noncanonical-missing-type",
        "http-400",
    ],
)
def test_malformed_or_wrong_status_route_response_fails(
    tmp_path: Path, code: str, body: str
) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    _write(fixtures / "endpoint.code", code + "\n")
    _write(fixtures / "endpoint.body", body)
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _state(result.stdout, "endpoint.reachable") == "FAIL"


def _run_existing_case(
    manifest: Path, fixtures: Path, baseline: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(PROBE),
            "--manifest",
            str(manifest),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            NOW,
        ],
        cwd=ROOT,
        env={**os.environ, "SN56_BASELINE_ENV": str(baseline)},
        text=True,
        capture_output=True,
        check=False,
    )


def test_wrong_validator_route_is_rejected_by_manifest(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["production"]["endpoint_route"] = "/training_repo/ImageTask"
    manifest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    result = _run_existing_case(manifest, fixtures, baseline)
    assert result.returncode == 2
    assert "production literals differ from fixed endpoint contract" in result.stderr
    assert "endpoint.reachable" not in result.stdout


def test_wrapper_forwards_manifest_target_without_sha_override(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    outdir = tmp_path / "wrapper-evidence"
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--manifest",
            str(manifest),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            NOW,
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SN56_BASELINE_ENV": str(baseline),
            "SN56_PROBE_OUTDIR": str(outdir),
            "SN56_PROBE_ATTEMPTS": "1",
            "SN56_PROBE_RETRY_DELAY": "0",
            "SN56_DISABLE_NOTIFICATION": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"manifest target {TARGET[:12]}" in result.stdout
    logs = list(outdir.glob("probe-*.log"))
    assert len(logs) == 1
    assert "RESULT: GREEN" in logs[0].read_text()


def test_wrapper_retries_the_same_private_manifest_snapshot(tmp_path: Path) -> None:
    manifest, fixtures, baseline = _green_case(tmp_path)
    policy = tmp_path / "docker-policy.json"
    readiness = tmp_path / "release-readiness.json"
    shutil.copyfile(DOCKER_POLICY_PATH, policy)
    shutil.copyfile(READINESS_PATH, readiness)
    _write(fixtures / "chain_uid.out", "UID=NONE\n")
    outdir = tmp_path / "wrapper-evidence"
    proc = subprocess.Popen(
        [
            "bash",
            str(WRAPPER),
            "--manifest",
            str(manifest),
            "--docker-policy",
            str(policy),
            "--readiness-receipt",
            str(readiness),
            "--mode",
            "mock",
            "--fixtures",
            str(fixtures),
            "--now",
            NOW,
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SN56_BASELINE_ENV": str(baseline),
            "SN56_PROBE_OUTDIR": str(outdir),
            "SN56_PROBE_ATTEMPTS": "2",
            "SN56_PROBE_RETRY_DELAY": "2",
            "SN56_DISABLE_NOTIFICATION": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    seen: list[str] = []
    while True:
        line = proc.stdout.readline()
        assert line, "wrapper exited before the retry boundary"
        seen.append(line)
        if "retrying in 2s" in line:
            break

    swapped = _manifest()
    swapped["target"]["commit"] = "1" * 40
    manifest.write_text(json.dumps(swapped), encoding="utf-8")
    policy.write_text("source policy replaced after snapshot\n", encoding="utf-8")
    readiness.write_text("source readiness replaced after snapshot\n", encoding="utf-8")
    _write(fixtures / "chain_uid.out", "UID=224\n")
    stdout_tail, stderr = proc.communicate(timeout=15)
    stdout = "".join(seen) + stdout_tail

    assert proc.returncode == 0, stdout + stderr
    result_files = list(outdir.glob("probe-*.json"))
    assert len(result_files) == 1
    receipt = json.loads(result_files[0].read_text())
    assert receipt["target_commit"] == TARGET
    assert f"target {TARGET[:12]}" in stdout
    assert ("1" * 12) not in stdout
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert '--docker-policy "$DOCKER_POLICY_SNAPSHOT"' in wrapper
    assert '--readiness-receipt "$READINESS_SNAPSHOT"' in wrapper
