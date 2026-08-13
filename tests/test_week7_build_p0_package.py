from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNC_PATH = ROOT / "ops" / "experiments" / "week7" / "safe_harvest_sync.py"
sync_spec = importlib.util.spec_from_file_location("safe_harvest_sync", SYNC_PATH)
sync = importlib.util.module_from_spec(sync_spec)
assert sync_spec.loader is not None
sys.modules[sync_spec.name] = sync
sync_spec.loader.exec_module(sync)

MODULE_PATH = ROOT / "ops" / "experiments" / "week7" / "build_p0_package.py"
module_spec = importlib.util.spec_from_file_location("build_p0_package", MODULE_PATH)
p0 = importlib.util.module_from_spec(module_spec)
assert module_spec.loader is not None
sys.modules[module_spec.name] = p0
module_spec.loader.exec_module(p0)

CAPTURE_PATH = ROOT / "ops" / "experiments" / "week7" / "capture_public_training_archives.py"
capture_spec = importlib.util.spec_from_file_location(
    "capture_public_training_archives_for_p0_test", CAPTURE_PATH
)
capture = importlib.util.module_from_spec(capture_spec)
assert capture_spec.loader is not None
sys.modules[capture_spec.name] = capture
capture_spec.loader.exec_module(capture)


TOURNAMENT = "tourn_aaaaaaaaaaaaaaaa_20260810"
TASK = "11111111-1111-4111-8111-111111111111"
HOTKEY = "5ExampLeFullHotkey"
REVISION = "1" * 40
NOW = dt.datetime(2026, 8, 10, 20, 0, tzinfo=dt.timezone.utc)


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def public_request_url(source: str, key: str) -> str:
    if source == "gradients-tournament":
        return f"https://api.gradients.io/tournament/{key}/details"
    if source == "gradients-task":
        return f"https://api.gradients.io/auditing/tasks/{key}"
    if source == "acceptance":
        return f"local://acceptance/{key}"
    if source == "hf-model":
        return f"https://huggingface.co/api/models/{key}"
    if source == "hf-revision-manifest":
        repo, revision = key.rsplit("/", 1)
        return f"https://huggingface.co/{repo}/tree/{revision}"
    if source == "hf-tree":
        repo, revision, _page = key.rsplit("/", 2)
        return f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    if source == "hf-file":
        parts = key.split("/")
        return (
            "https://huggingface.co/api/resolve-cache/models/"
            + "/".join(parts[:2])
            + "/"
            + parts[2]
            + "/"
            + "/".join(parts[3:])
        )
    return f"https://example.invalid/{key}"


def write(root: Path, relative: str, body: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def cas(root: Path, body: bytes) -> str:
    digest = sha(body)
    write(root, f"objects/sha256/{digest[:2]}/{digest}", body)
    return digest


def safe_api(
    tmp_path: Path,
    *,
    tournament_type: str = "image",
    round_status: str = "completed",
    task_status: str = "completed",
    task_loss: float = 0.05,
    round_loss: float = 0.05,
) -> Path:
    root = tmp_path / "api"
    root.mkdir()
    repo = f"gradients-io-tournaments/tournament-{TOURNAMENT}-{TASK}-5ExampLe"
    task_score = {
        "hotkey": HOTKEY,
        "repo": repo,
        "submission_id": "submission-1",
        "test_loss": task_loss,
        "synth_loss": task_loss,
        "quality_score": 1.0,
        "rank": 1,
        "score_reason": None,
        "derived_state": "scored",
    }
    round_score = {**task_score, "test_loss": round_loss, "synth_loss": round_loss}
    value = {
        "schema": "sn56.week7.safe-public-api-snapshot",
        "schema_version": 1,
        "observed_at": "2026-08-10T19:59:00Z",
        "tournament": {
            "tournament_id": TOURNAMENT,
            "tournament_type": tournament_type,
            "status": "active",
            "participants": [{"hotkey": HOTKEY}],
            "rounds": [
                {
                    "round_id": "round-1",
                    "round_number": 1,
                    "round_type": "group",
                    "status": round_status,
                    "participants": [HOTKEY],
                    "tasks": [
                        {
                            "task_id": TASK,
                            "task_type": "ImageTask",
                            "winner": HOTKEY,
                            "participant_scores": [round_score],
                        }
                    ],
                }
            ],
        },
        "tasks": [
            {
                "task_id": TASK,
                "task_type": "ImageTask",
                "model_id": "krea/Krea-2-Raw",
                "model_type": "krea2",
                "hours_to_complete": 0.75,
                "status": task_status,
                "public_training_archive_present": True,
                "participants": [task_score],
            }
        ],
        "source_responses": [
            {
                "url": f"https://api.gradients.io/tournament/{TOURNAMENT}/details",
                "response_sha256": "a" * 64,
                "response_bytes": 100,
            },
            {
                "url": f"https://api.gradients.io/auditing/tasks/{TASK}",
                "response_sha256": "b" * 64,
                "response_bytes": 100,
            },
        ],
    }
    body = sync.canonical_json(value)
    stamp = "20260810T195900.000000Z"
    write(root, f"snapshots/{stamp}.json", body)
    write(root, f"snapshots/{stamp}.sha256", f"{sha(body)}  {stamp}.json\n".encode())
    return root / f"snapshots/{stamp}.json"


def raw_header(step: int) -> tuple[bytes, dict]:
    parsed = {
        "__metadata__": {"training_info": json.dumps({"step": step, "epoch": 1})},
        "layer.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
    }
    payload = json.dumps(parsed, separators=(",", ":")).encode()
    return struct.pack("<Q", len(payload)) + payload, parsed


def raw_watcher(
    tmp_path: Path,
    *,
    truncated: bool = False,
    terminal_round_status: str = "completed",
    terminal_tournament_type: str = "image",
    terminal_r1_task: str = TASK,
    later_round_task: str | None = None,
    extra_observation_source: str | None = None,
    extra_tree_revision: str | None = None,
    omit_model_head: bool = False,
    omit_config_from_manifest: bool = False,
    ghost_capture: bool = False,
    eligible_labels: tuple[str, ...] = ("last", "800", "1000"),
) -> tuple[Path, dict[str, dict]]:
    root = tmp_path / "watcher"
    root.mkdir()
    write(
        root,
        "ROOT-IDENTITY.json",
        sync.canonical_json(
            {
                "schema": "sn56.week7.harvest-root-identity",
                "schema_version": 1,
                "source": "fixture",
                "tournament_id": TOURNAMENT,
            }
        ),
    )
    repo = f"gradients-io-tournaments/tournament-{TOURNAMENT}-{TASK}-5ExampLe"
    checkpoint_oids = {"last": "2" * 64, "800": "2" * 64, "1000": "3" * 64}
    checkpoint_sizes = {
        label: len(raw_header(800 if label in {"last", "800"} else 1000)[0]) + 4
        for label in checkpoint_oids
    }
    config = b"""job: extension
config:
  process:
    - network: {type: lora, linear: 32, linear_alpha: 32}
      save: {dtype: bf16, save_every: 200}
      datasets:
        - {caption_dropout_rate: 0.05, cache_latents_to_disk: true, resolution: [512, 768, 1024]}
      train:
        steps: 1000
        lr: 0.0001
        optimizer: adamw8bit
        train_text_encoder: false
        ema_config: {use_ema: true, ema_decay: 0.995}
        optimizer_params: {weight_decay: 0.01}
        lr_scheduler_params: {min_lr_ratio: 0.1}
      model: {arch: krea2, quantize: false}
"""
    config_sha = cas(root, config)
    forge_run = sync.canonical_json(
        {
            "kind": "forge-public-run-recorder",
            "schema": 2,
            "private_record_sha256": "9" * 64,
            "events": [{"name": "run_complete", "t": 123.4}],
        }
    )
    forge_run_sha = cas(root, forge_run)
    tree = sync.canonical_json(
        [
            {"type": "file", "path": "checkpoints/config.yaml", "size": len(config)},
            {"type": "file", "path": "checkpoints/forge_run.json", "size": len(forge_run)},
            *[
                {
                    "type": "file",
                    "path": f"checkpoints/{'last' if name == 'last' else 'last_' + name}.safetensors",
                    "size": checkpoint_sizes[name],
                    "lfs": {"oid": oid, "size": checkpoint_sizes[name]},
                }
                for name, oid in checkpoint_oids.items()
            ],
        ]
    )
    manifest = sync.canonical_json(
        {
            "capture_complete": True,
            "processing_complete": True,
            "tree_truncated": truncated,
            "tree_entry_count": 5,
            "tree_file_count": 5,
            "captures": [
                {
                    "path": "checkpoints/config.yaml",
                    "kind": "small",
                    "captured": True,
                    "object_sha256": config_sha,
                    "declared_size": len(config),
                    "bytes": len(config),
                },
                {
                    "path": "checkpoints/forge_run.json",
                    "kind": "small",
                    "captured": True,
                    "object_sha256": forge_run_sha,
                    "bytes": len(forge_run),
                    "declared_size": len(forge_run),
                },
                *(
                    [
                        {
                            "path": "ghost/forge_run.json",
                            "kind": "small",
                            "captured": True,
                            "object_sha256": forge_run_sha,
                            "bytes": len(forge_run),
                            "declared_size": len(forge_run),
                        }
                    ]
                    if ghost_capture
                    else []
                ),
            ],
            "config_absent": omit_config_from_manifest,
            "configs": [] if omit_config_from_manifest else ["checkpoints/config.yaml"],
            "eligible_weight_plan": [
                {
                    "path": f"checkpoints/{'last' if name == 'last' else 'last_' + name}.safetensors",
                    "lfs_oid": oid,
                    "size": checkpoint_sizes[name],
                }
                for name, oid in checkpoint_oids.items()
                if name in eligible_labels
            ],
            "failures": [],
            "skipped": [],
            "repo_id": repo,
            "revision": REVISION,
        }
    )
    model = sync.canonical_json({"id": repo, "modelId": repo, "sha": REVISION})

    files: list[dict] = []

    def observation(source: str, key: str, content: bytes) -> tuple[str, str]:
        digest = cas(root, content)
        wrapper = sync.canonical_json(
            {
                "schema": 1,
                "source": source,
                "key": key,
                "request_url": public_request_url(source, key),
                "status": 200,
                "observed_at": "2026-08-10T19:55:00Z",
                "content_sha256": digest,
                "content_bytes": len(content),
                "object": f"objects/sha256/{digest[:2]}/{digest}",
            }
        )
        relative = (
            f"observations/{source}/{key}/"
            f"20260810T195500.000000Z-{digest[:12]}.json"
        )
        write(root, relative, wrapper)
        return relative, sha(wrapper)

    tree_key = f"{repo}/{REVISION}/page-0001"
    tree_path, tree_wrapper_sha = observation("hf-tree", tree_key, tree)
    if extra_tree_revision is not None:
        observation("hf-tree", f"{repo}/{extra_tree_revision}/page-0001", tree)
    if not omit_model_head:
        observation("hf-model", repo, model)
    observation("hf-revision-manifest", f"{repo}/{REVISION}", manifest)
    observation(
        "hf-file",
        f"{repo}/{REVISION}/checkpoints/config.yaml",
        sync.canonical_json(
            {
                "repo_id": repo,
                "revision": REVISION,
                "path": "checkpoints/config.yaml",
                "kind": "small",
                "declared_size": len(config),
                "bytes": len(config),
                "object_sha256": config_sha,
            }
        ),
    )
    observation(
        "hf-file",
        f"{repo}/{REVISION}/checkpoints/forge_run.json",
        sync.canonical_json(
            {
                "repo_id": repo,
                "revision": REVISION,
                "path": "checkpoints/forge_run.json",
                "kind": "small",
                "declared_size": len(forge_run),
                "bytes": len(forge_run),
                "object_sha256": forge_run_sha,
            }
        ),
    )
    tournament_rounds = [
        {
            "round_number": 1,
            "status": terminal_round_status,
            "tasks": [{"task_id": terminal_r1_task}],
        }
    ]
    if later_round_task is not None:
        tournament_rounds.append(
            {
                "round_number": 2,
                "status": "active",
                "tasks": [{"task_id": later_round_task}],
            }
        )
    tournament_body = sync.canonical_json(
        {
            "tournament_id": TOURNAMENT,
            "tournament_type": terminal_tournament_type,
            "rounds": tournament_rounds,
        }
    )
    tournament_path, _ = observation(
        "gradients-tournament", TOURNAMENT, tournament_body
    )
    tournament_digest = json.loads((root / tournament_path).read_text())["content_sha256"]
    observation(
        "gradients-task",
        TASK,
        sync.canonical_json({"task_id": TASK, "status": "completed"}),
    )
    if extra_observation_source is not None:
        observation(
            extra_observation_source,
            f"{repo}/{REVISION}/page-0002",
            sync.canonical_json({"public": "metadata"}),
        )

    # The exact sync ledger binds every selected observation and every CAS,
    # including the fixture bytes needed for in-memory byte de-duplication.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "ROOT-IDENTITY.json":
            continue
        relative = path.relative_to(root).as_posix()
        body = path.read_bytes()
        files.append({"path": relative, "sha256": sha(body), "bytes": len(body), "created": True})
    ledger = {
        "schema": "sn56.week7.safe-harvest-sync",
        "schema_version": 2,
        "complete": True,
        "status": "COMPLETE",
        "errors": [],
        "observed_at": "2026-08-10T20:00:00Z",
        "root_identity_sha256": sha((root / "ROOT-IDENTITY.json").read_bytes()),
        "scope": {
            "tournament_id": TOURNAMENT,
            "task_ids": [TASK],
            "derived_from_observation": tournament_path,
            "derived_from_content_sha256": tournament_digest,
        },
        "files": files,
    }
    ledger_body = sync.canonical_json(ledger)
    stamp = "20260810T200000.000000Z"
    write(root, f"ledgers/{stamp}.json", ledger_body)
    write(root, f"ledgers/{stamp}.sha256", f"{sha(ledger_body)}  {stamp}.json\n".encode())
    return root, {
        "repo": repo,
        "tree_path": tree_path,
        "tree_wrapper_sha": tree_wrapper_sha,
        "tree_content_sha": json.loads((root / tree_path).read_text())["content_sha256"],
        "checkpoint_oids": checkpoint_oids,
        "checkpoint_sizes": checkpoint_sizes,
    }


def header_root(tmp_path: Path, watcher: Path, context: dict) -> tuple[Path, Path]:
    root = tmp_path / "headers"
    root.mkdir()
    tree_wrapper = json.loads((watcher / context["tree_path"]).read_bytes())
    tree_observed_at = sync.observation_timestamp(context["tree_path"], tree_wrapper)
    write(
        root,
        "ROOT-IDENTITY.json",
        sync.canonical_json(
            {
                "schema": "sn56.week7.public-safetensors-header-root",
                "schema_version": 1,
                "input_root": str(watcher.absolute()),
                "tournament_id": TOURNAMENT,
                "task_ids": [TASK],
            }
        ),
    )
    associations = []
    for label, oid in context["checkpoint_oids"].items():
        step = 800 if label in {"last", "800"} else 1000
        header_body, parsed = raw_header(step)
        header_sha = cas(root, header_body)
        record = {
            "schema": "sn56.week7.public-safetensors-headers",
            "schema_version": 1,
            "lfs_sha256": oid,
            "lfs_bytes": context["checkpoint_sizes"][label],
            "header_sha256": header_sha,
            "header_bytes": len(header_body),
            "header_object": f"objects/sha256/{header_sha[:2]}/{header_sha}",
            "metadata": parsed["__metadata__"],
            "tensor_count": 1,
            "tensors": [
                {
                    "name": "layer.weight",
                    "dtype": "F16",
                    "shape": [2],
                    "data_offsets": [0, 4],
                }
            ],
        }
        record_body = sync.canonical_json(record)
        record_path = f"records/{oid[:2]}/{oid}.json"
        if not (root / record_path).exists():
            write(root, record_path, record_body)
        path = f"checkpoints/{'last' if label == 'last' else 'last_' + label}.safetensors"
        associations.append(
            {
                "repository": context["repo"],
                "revision": REVISION,
                "task_id": TASK,
                "path": path,
                "lfs_sha256": oid,
                "lfs_bytes": context["checkpoint_sizes"][label],
                "record": record_path,
                "record_sha256": sha(record_body),
                "header_sha256": header_sha,
                "tree_observation": context["tree_path"],
                "tree_observation_sha256": context["tree_wrapper_sha"],
                "tree_content_sha256": context["tree_content_sha"],
                "tree_observations": [
                    {
                        "observed_at": tree_observed_at,
                        "tree_observation": context["tree_path"],
                        "tree_observation_sha256": context["tree_wrapper_sha"],
                        "tree_content_sha256": context["tree_content_sha"],
                    }
                ],
            }
        )
    inventory = {
        "schema": "sn56.week7.public-safetensors-headers.inventory",
        "schema_version": 1,
        "observed_at": "2026-08-10T19:59:00Z",
        "tournament_id": TOURNAMENT,
        "task_ids": [TASK],
        "candidate_count": len(associations),
        "unique_lfs_objects": len(set(context["checkpoint_oids"].values())),
        "associations": associations,
    }
    path = root / "inventories/20260810T195900.000000Z.json"
    write(root, path.relative_to(root).as_posix(), sync.canonical_json(inventory))
    return root, path


def inputs(tmp_path: Path, *, truncated: bool = False):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, truncated=truncated)
    headers, inventory = header_root(tmp_path, watcher, context)
    return api, watcher, headers, inventory


def rewrite_safe_api(path: Path, mutator) -> None:
    value = json.loads(path.read_bytes())
    mutator(value)
    body = sync.canonical_json(value)
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(
        f"{sha(body)}  {path.name}\n", encoding="ascii"
    )


def rewrite_watcher_file_and_ledger(root: Path, relative: str, body: bytes) -> None:
    """Keep a test watcher's synthetic COMPLETE ledger bound to one mutation."""
    write(root, relative, body)
    ledger_path = next((root / "ledgers").glob("*.json"))
    ledger = json.loads(ledger_path.read_bytes())
    rows = ledger["files"]
    replacement = {
        "path": relative,
        "sha256": sha(body),
        "bytes": len(body),
        "created": True,
    }
    for index, row in enumerate(rows):
        if row["path"] == relative:
            rows[index] = replacement
            break
    else:
        rows.append(replacement)
    ledger_body = sync.canonical_json(ledger)
    ledger_path.write_bytes(ledger_body)
    ledger_path.with_suffix(".sha256").write_text(
        f"{sha(ledger_body)}  {ledger_path.name}\n", encoding="ascii"
    )


def training_archive_inputs(
    tmp_path: Path,
    api: Path,
    *,
    partial: bool = False,
) -> tuple[Path, Path, Path, Path | None]:
    root = tmp_path / "training-archives"
    root.mkdir()
    write(
        root,
        "ROOT-IDENTITY.json",
        sync.canonical_json(
            {
                "schema": p0.TRAINING_ARCHIVE_ROOT_SCHEMA,
                "schema_version": p0.TRAINING_ARCHIVE_SCHEMA_VERSION,
                "tournament_id": TOURNAMENT,
                "authority": "exact completed-R1 public training_data URLs only",
            }
        ),
    )

    stamp = "20260810T200000.000000Z"
    receipt_path = root / f"receipts/{TASK}/{stamp}.json"
    archive_path: Path | None = None
    base_receipt = {
        "schema": p0.TRAINING_ARCHIVE_RECEIPT_SCHEMA,
        "schema_version": p0.TRAINING_ARCHIVE_SCHEMA_VERSION,
        "observed_at": sync.utc_iso(NOW),
        "tournament_id": TOURNAMENT,
        "task_id": TASK,
        "task_status": "completed",
        "source_task_endpoint": f"https://api.gradients.io/auditing/tasks/{TASK}",
        "source_response_sha256": "4" * 64,
        "source_response_bytes": 128,
        "input_safe_api_snapshot_sha256": sha(api.read_bytes()),
        "rights": p0.TRAINING_ARCHIVE_RIGHTS,
        "exclusion_contract": p0.TRAINING_ARCHIVE_EXCLUSION_CONTRACT,
    }
    if partial:
        receipt = {
            **base_receipt,
            "status": "PARTIAL",
            "reason": "public training_data URL absent",
            "archive": None,
            "inventory": None,
        }
    else:
        payloads = [
            ("dataset/000.png", b"first-image"),
            ("dataset/000.txt", b"first caption"),
            ("dataset/001.png", b"second-image"),
            ("dataset/001.txt", b"second caption"),
            ("dataset/002.png", b"first-image"),
            ("dataset/002.txt", b"duplicate caption"),
        ]
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, body in payloads:
                archive.writestr(name, body)
        archive_body = output.getvalue()
        archive_sha = cas(root, archive_body)
        archive_path = root / f"objects/sha256/{archive_sha[:2]}/{archive_sha}"
        members = []
        with zipfile.ZipFile(io.BytesIO(archive_body), "r") as archive:
            for index, info in enumerate(archive.infolist()):
                body = archive.read(info)
                members.append(
                    {
                        "central_directory_index": index,
                        "path": info.filename,
                        "kind": "image" if info.filename.endswith(".png") else "caption",
                        "uncompressed_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "sha256": sha(body),
                        "crc32": f"{info.CRC:08x}",
                        "compression_method": info.compress_type,
                    }
                )
        receipt = {
            **base_receipt,
            "status": "COMPLETE",
            "reason": None,
            "source_training_archive": {
                "url": "https://s3.eu-central-003.backblazeb2.com/training.zip",
                "http_status": 200,
                "content_type": "application/zip",
                "declared_content_length": len(archive_body),
            },
            "archive": {
                "sha256": archive_sha,
                "bytes": len(archive_body),
                "object": f"objects/sha256/{archive_sha[:2]}/{archive_sha}",
            },
            "inventory": {
                "member_count": len(members),
                "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in members),
                "image_member_count": 3,
                "caption_member_count": 3,
                "paired_stem_count": 3,
                "post_exact_byte_dedup_image_count": 2,
                "members": members,
            },
        }
    receipt_body = sync.canonical_json(receipt)
    write(root, receipt_path.relative_to(root).as_posix(), receipt_body)

    task_row = {
        "task_id": TASK,
        "status": receipt["status"],
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": sha(receipt_body),
        "archive_sha256": receipt["archive"]["sha256"] if receipt["archive"] else None,
        "image_member_count": receipt["inventory"]["image_member_count"]
        if receipt["inventory"]
        else None,
        "post_exact_byte_dedup_image_count": receipt["inventory"][
            "post_exact_byte_dedup_image_count"
        ]
        if receipt["inventory"]
        else None,
    }
    inventory = {
        "schema": p0.TRAINING_ARCHIVE_INVENTORY_SCHEMA,
        "schema_version": p0.TRAINING_ARCHIVE_SCHEMA_VERSION,
        "observed_at": sync.utc_iso(NOW),
        "status": receipt["status"],
        "tournament_id": TOURNAMENT,
        "round_number": 1,
        "round_status": "completed",
        "input_safe_api_snapshot": {"filename": api.name, "sha256": sha(api.read_bytes())},
        "task_count": 1,
        "complete_task_count": 0 if partial else 1,
        "partial_task_count": 1 if partial else 0,
        "tasks": [task_row],
        "rights": p0.TRAINING_ARCHIVE_RIGHTS,
    }
    inventory_path = root / f"inventories/{stamp}.json"
    inventory_body = sync.canonical_json(inventory)
    write(root, inventory_path.relative_to(root).as_posix(), inventory_body)
    checksum_path = inventory_path.with_suffix(".sha256")
    write(
        root,
        checksum_path.relative_to(root).as_posix(),
        f"{sha(inventory_body)}  {inventory_path.name}\n".encode(),
    )
    return root, inventory_path, checksum_path, archive_path


def rewrite_training_inventory(inventory_path: Path, value: dict) -> None:
    body = sync.canonical_json(value)
    inventory_path.write_bytes(body)
    inventory_path.with_suffix(".sha256").write_text(
        f"{sha(body)}  {inventory_path.name}\n", encoding="ascii"
    )


def rewrite_training_receipt(
    training_root: Path, inventory_path: Path, mutate
) -> None:
    inventory = json.loads(inventory_path.read_bytes())
    task_row = inventory["tasks"][0]
    receipt_path = training_root / task_row["receipt"]
    receipt = json.loads(receipt_path.read_bytes())
    mutate(receipt)
    body = sync.canonical_json(receipt)
    receipt_path.write_bytes(body)
    task_row["receipt_sha256"] = sha(body)
    rewrite_training_inventory(inventory_path, inventory)


def run_build(tmp_path: Path, *, truncated: bool = False):
    api, watcher, headers, inventory = inputs(tmp_path, truncated=truncated)
    output = tmp_path / "output"
    result = p0.build_package(
        api,
        watcher,
        headers,
        inventory,
        output,
        TOURNAMENT,
        observed_at=NOW,
    )
    package = json.loads((output / result["package"]).read_text())
    return result, package, (api, watcher, headers, inventory, output)


def test_package_joins_scores_artifacts_and_marks_missing_training_archive(tmp_path):
    result, package, _ = run_build(tmp_path)

    assert result["status"] == "PARTIAL"
    participant = package["round"]["tasks"][0]["participants"][0]
    assert participant["derived_state"] == "scored"
    assert participant["task_api"]["test_loss"] == 0.05
    revision = package["public_repositories"][0]["revisions"][0]
    recipe = revision["configs"][0]["recipe_projection"]["fields"]
    assert recipe["config.process[0].train.steps"] == 1000
    assert recipe["config.process[0].train.ema_config.use_ema"] is True
    assert recipe["config.process[0].train.optimizer_params.weight_decay"] == 0.01
    assert recipe["config.process[0].train.lr_scheduler_params.min_lr_ratio"] == 0.1
    assert any(row["path"] == "checkpoints/forge_run.json" for row in revision["file_tree"])
    assert revision["public_run_telemetry"][0]["projection"]["events"] == [
        {"name": "run_complete", "t": 123.4}
    ]
    comparison = revision["checkpoint_identity"]["last_vs_numbered"][0]
    assert comparison["submitted_artifact_step_from_header"] == 800
    assert comparison["byte_identical_numbered_steps"] == [800]
    assert comparison["highest_observed_numbered_step"] == 1000
    assert comparison["classification"] == "last-matches-earlier-numbered-checkpoint"
    fixture = package["public_training_archive_inventory"][0]
    assert fixture["actual_train_zip_image_count"] is None
    assert any(
        row["class"] == "public-training-archive-counts-unavailable"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )
    serialized = json.dumps(package)
    assert "optimizer_visible_image_count" not in serialized


def test_real_sync_ledger_with_policy_vocabulary_is_consumable(tmp_path):
    api = safe_api(tmp_path)
    source, context = raw_watcher(tmp_path)
    synced = tmp_path / "synced-watcher"
    sync_result = sync.sync_archive(
        sync.LocalSource(source),
        synced,
        observed_at=dt.datetime(2026, 8, 10, 20, 1, tzinfo=dt.timezone.utc),
        tournament_id=TOURNAMENT,
    )
    assert sync_result["status"] == "COMPLETE"
    ledger = (synced / sync_result["ledger_path"]).read_bytes()
    assert sync.contains_forbidden_body(ledger) is True
    headers, inventory = header_root(tmp_path, synced, context)

    result = p0.build_package(
        api,
        synced,
        headers,
        inventory,
        tmp_path / "output",
        TOURNAMENT,
        observed_at=NOW,
    )

    # Training archive evidence is intentionally absent in this integration
    # test; the producer/consumer ledger path itself must still reconcile.
    assert result["status"] == "PARTIAL"
    assert result["public_repository_count"] == 1


@pytest.mark.parametrize(
    ("tournament_type", "round_status", "message"),
    [("text", "completed", "not an image"), ("image", "active", "not completed")],
)
def test_non_image_or_unfinished_round_fails_before_packaging(
    tmp_path, tournament_type, round_status, message
):
    api = safe_api(tmp_path, tournament_type=tournament_type, round_status=round_status)
    watcher, context = raw_watcher(tmp_path)
    headers, inventory = header_root(tmp_path, watcher, context)
    with pytest.raises(sync.IntegrityError, match=message):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
        )


def test_safe_api_source_endpoint_provenance_is_mandatory(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    rewrite_safe_api(api, lambda value: value.__setitem__("source_responses", []))
    with pytest.raises(sync.IntegrityError, match="exact endpoint set"):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
        )


def test_p0_rejects_raw_watcher_query_even_when_complete_ledger_rebinds_it(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    wrapper_path = next((watcher / "observations" / "gradients-task").rglob("*.json"))
    relative = wrapper_path.relative_to(watcher).as_posix()
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["request_url"] += "?token=must-not-persist"
    rewrite_watcher_file_and_ledger(watcher, relative, sync.canonical_json(wrapper))

    with pytest.raises(sync.IntegrityError, match="query"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output-query",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_p0_rejects_numeric_suffix_path_named_by_complete_ledger(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    relative = "snapshots/20260810T195900.000000Z/test1.json"
    rewrite_watcher_file_and_ledger(watcher, relative, b'{"public":"metadata"}\n')

    with pytest.raises(sync.IntegrityError, match="prohibited path"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output-suffix",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_safe_api_schema_bool_and_filename_timestamp_fail_closed(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    rewrite_safe_api(api, lambda value: value.__setitem__("schema_version", True))
    with pytest.raises(sync.IntegrityError, match="schema is invalid"):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output-a", TOURNAMENT, observed_at=NOW
        )

    second = tmp_path / "second"
    second.mkdir()
    api = safe_api(second)
    watcher, context = raw_watcher(second)
    headers, inventory = header_root(second, watcher, context)
    rewrite_safe_api(
        api, lambda value: value.__setitem__("observed_at", "2026-08-10T19:58:59Z")
    )
    with pytest.raises(sync.IntegrityError, match="snapshot filename"):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output-b", TOURNAMENT, observed_at=NOW
        )


def test_score_schema_and_derived_state_are_rechecked_at_consumption():
    with pytest.raises(sync.IntegrityError, match="invalid test_loss"):
        p0._score_row_keyed(
            [{"hotkey": "h", "test_loss": "0.05", "derived_state": "scored"}],
            "fixture",
        )
    with pytest.raises(sync.IntegrityError, match="invalid rank"):
        p0._score_row_keyed(
            [{"hotkey": "h", "rank": True, "derived_state": "pending"}],
            "fixture",
        )
    with pytest.raises(sync.IntegrityError, match="invalid derived_state"):
        p0._score_row_keyed(
            [{"hotkey": "h", "test_loss": 0.05, "derived_state": "pending"}],
            "fixture",
        )


def test_participant_repository_suffix_must_match_hotkey_prefix():
    repo = f"gradients-io-tournaments/tournament-{TOURNAMENT}-{TASK}-5ExampLe"
    assert p0._repo_from_api(repo, TOURNAMENT, TASK, HOTKEY) == repo
    with pytest.raises(sync.IntegrityError, match="contradicts its hotkey"):
        p0._repo_from_api(repo, TOURNAMENT, TASK, "5DifferentHotkey")


def test_nonterminal_task_detail_is_reported_partial(tmp_path):
    api = safe_api(tmp_path, task_status="training")
    watcher, context = raw_watcher(tmp_path)
    headers, inventory = header_root(tmp_path, watcher, context)

    result = p0.build_package(
        api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
    )
    package = json.loads((tmp_path / "output" / result["package"]).read_text())

    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "task-api-not-terminal"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_task_and_round_score_conflict_is_reported_partial(tmp_path):
    api = safe_api(tmp_path, task_loss=0.05, round_loss=0.07)
    watcher, context = raw_watcher(tmp_path)
    headers, inventory = header_root(tmp_path, watcher, context)

    result = p0.build_package(
        api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
    )
    package = json.loads((tmp_path / "output" / result["package"]).read_text())

    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "public-api-surface-conflict" and "test_loss" in row["fields"]
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_watcher_scope_must_itself_bind_completed_round_one(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, terminal_round_status="active")
    headers, inventory = header_root(tmp_path, watcher, context)

    with pytest.raises(sync.IntegrityError, match="not bound to a completed Round 1"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_watcher_terminal_body_must_assert_selected_image_tournament(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, terminal_tournament_type="text")
    headers, inventory = header_root(tmp_path, watcher, context)

    with pytest.raises(sync.IntegrityError, match="tournament body contradicts"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_watcher_ledger_rejects_unknown_hf_observation_source(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, extra_observation_source="hf-secret")
    headers, inventory = header_root(tmp_path, watcher, context)

    with pytest.raises(sync.IntegrityError, match="out-of-scope observation"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_watcher_ledger_rejects_unreferenced_opaque_cas(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path)
    extra = b"opaque bytes that no allowed observation references"
    extra_digest = cas(watcher, extra)
    ledger_path = next((watcher / "ledgers").glob("*.json"))
    ledger = json.loads(ledger_path.read_bytes())
    ledger["files"].append(
        {
            "path": f"objects/sha256/{extra_digest[:2]}/{extra_digest}",
            "sha256": extra_digest,
            "bytes": len(extra),
            "created": True,
        }
    )
    ledger_body = sync.canonical_json(ledger)
    ledger_path.write_bytes(ledger_body)
    ledger_path.with_suffix(".sha256").write_text(
        f"{sha(ledger_body)}  {ledger_path.name}\n", encoding="ascii"
    )
    headers, inventory = header_root(tmp_path, watcher, context)

    with pytest.raises(sync.IntegrityError, match="unreferenced CAS"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_newer_partial_watcher_ledger_blocks_older_complete_ledger(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    old_ledger_path = next((watcher / "ledgers").glob("*.json"))
    partial = json.loads(old_ledger_path.read_bytes())
    partial["complete"] = False
    partial["status"] = "PARTIAL"
    partial["errors"] = ["simulated terminal sync failure"]
    partial["observed_at"] = "2026-08-10T20:01:00Z"
    body = sync.canonical_json(partial)
    name = "20260810T200100.000000Z"
    write(watcher, f"ledgers/{name}.json", body)
    write(watcher, f"ledgers/{name}.sha256", f"{sha(body)}  {name}.json\n".encode())

    with pytest.raises(sync.IntegrityError, match="newest raw-watcher ledger is not COMPLETE"):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
        )


def test_ledger_body_time_must_match_checksum_bound_filename(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    ledger_path = next((watcher / "ledgers").glob("*.json"))
    ledger = json.loads(ledger_path.read_bytes())
    ledger["observed_at"] = "2026-08-10T20:02:00Z"
    body = sync.canonical_json(ledger)
    ledger_path.write_bytes(body)
    ledger_path.with_suffix(".sha256").write_text(
        f"{sha(body)}  {ledger_path.name}\n", encoding="ascii"
    )

    with pytest.raises(sync.IntegrityError, match="does not match its filename"):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
        )


def test_watcher_terminal_round_one_must_match_exact_api_task_membership(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(
        tmp_path,
        terminal_r1_task="22222222-2222-4222-8222-222222222222",
        later_round_task=TASK,
    )
    headers, inventory = header_root(tmp_path, watcher, context)

    with pytest.raises(sync.IntegrityError, match="task membership mismatches"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_header_root_must_bind_the_selected_watcher_root(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    identity_path = headers / "ROOT-IDENTITY.json"
    identity = json.loads(identity_path.read_bytes())
    identity["input_root"] = str((tmp_path / "different-watcher").absolute())
    identity_path.write_bytes(sync.canonical_json(identity))

    with pytest.raises(sync.IntegrityError, match="different watcher root"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_header_root_task_allowlist_must_equal_exact_round_one(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    identity_path = headers / "ROOT-IDENTITY.json"
    identity = json.loads(identity_path.read_bytes())
    identity["task_ids"] = ["22222222-2222-4222-8222-222222222222"]
    identity_path.write_bytes(sync.canonical_json(identity))

    with pytest.raises(sync.IntegrityError, match="task allowlist mismatches"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_header_association_without_tree_provenance_fails_closed(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    value = json.loads(inventory.read_bytes())
    value["associations"][0].pop("tree_observations")
    inventory.write_bytes(sync.canonical_json(value))

    with pytest.raises(sync.IntegrityError, match="lacks tree provenance"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_header_tree_provenance_timestamp_must_match_hash_verified_wrapper(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    value = json.loads(inventory.read_bytes())
    value["associations"][0]["tree_observations"][0]["observed_at"] = (
        "2999-01-01T00:00:00Z"
    )
    inventory.write_bytes(sync.canonical_json(value))

    with pytest.raises(sync.IntegrityError, match="timestamp is inconsistent"):
        p0.build_package(
            api,
            watcher,
            headers,
            inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
        )


def test_tampered_header_record_hash_fails_closed(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    record = next((headers / "records").rglob("*.json"))
    record.write_text("{}\n")
    with pytest.raises(sync.IntegrityError, match="record hash mismatch"):
        p0.build_package(
            api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
        )


def test_full_pool_fixture_symlink_is_out_of_scope_and_never_followed(tmp_path):
    api, watcher, headers, inventory = inputs(tmp_path)
    bad = watcher / "observations/fixtures" / TASK / "hidden" / "test_rows.json"
    bad.parent.mkdir(parents=True)
    bad.symlink_to(tmp_path / "does-not-exist")
    result = p0.build_package(
        api, watcher, headers, inventory, tmp_path / "output", TOURNAMENT, observed_at=NOW
    )
    assert result["status"] == "PARTIAL"


def test_truncated_repository_capture_is_reported_partial_not_silently_complete(tmp_path):
    result, package, _ = run_build(tmp_path, truncated=True)
    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "repository-revision-capture-partial-or-truncated"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_tree_revision_without_manifest_is_fail_visible(tmp_path):
    api = safe_api(tmp_path)
    extra_revision = "4" * 40
    watcher, context = raw_watcher(tmp_path, extra_tree_revision=extra_revision)
    headers, inventory = header_root(tmp_path, watcher, context)

    result = p0.build_package(
        api,
        watcher,
        headers,
        inventory,
        tmp_path / "output",
        TOURNAMENT,
        observed_at=NOW,
    )
    package = json.loads((tmp_path / "output" / result["package"]).read_bytes())

    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "tree-revision-has-no-manifest"
        and row["revision"] == extra_revision
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_public_repository_without_model_head_is_fail_visible(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, omit_model_head=True)
    headers, inventory = header_root(tmp_path, watcher, context)

    result = p0.build_package(
        api,
        watcher,
        headers,
        inventory,
        tmp_path / "output",
        TOURNAMENT,
        observed_at=NOW,
    )
    package = json.loads((tmp_path / "output" / result["package"]).read_bytes())

    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "public-repository-has-no-model-head"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_tree_checkpoint_omitted_from_eligible_plan_is_still_reported(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, eligible_labels=("last", "800"))
    headers, inventory = header_root(tmp_path, watcher, context)
    output = tmp_path / "output"

    result = p0.build_package(
        api, watcher, headers, inventory, output, TOURNAMENT, observed_at=NOW
    )
    package = json.loads((output / result["package"]).read_text())
    checkpoints = package["public_repositories"][0]["revisions"][0][
        "checkpoint_identity"
    ]["checkpoints"]

    assert result["status"] == "PARTIAL"
    assert any(row["path"].endswith("last_1000.safetensors") for row in checkpoints)
    assert any(
        row["class"] == "tree-checkpoint-omitted-from-eligible-plan"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_tree_config_omitted_from_manifest_is_fail_visible(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, omit_config_from_manifest=True)
    headers, inventory = header_root(tmp_path, watcher, context)
    output = tmp_path / "output"

    result = p0.build_package(
        api, watcher, headers, inventory, output, TOURNAMENT, observed_at=NOW
    )
    package = json.loads((output / result["package"]).read_text())

    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "tree-config-omitted-from-manifest"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_manifest_capture_absent_from_tree_is_fail_visible(tmp_path):
    api = safe_api(tmp_path)
    watcher, context = raw_watcher(tmp_path, ghost_capture=True)
    headers, inventory = header_root(tmp_path, watcher, context)
    output = tmp_path / "output"
    result = p0.build_package(
        api, watcher, headers, inventory, output, TOURNAMENT, observed_at=NOW
    )
    package = json.loads((output / result["package"]).read_text())

    assert result["status"] == "PARTIAL"
    assert any(
        row["class"] == "manifest-capture-absent-from-immutable-tree"
        and row["path"] == "ghost/forge_run.json"
        for row in package["completeness"]["missing_or_partial_surfaces"]
    )


def test_tree_file_size_must_equal_lfs_size():
    with pytest.raises(sync.IntegrityError, match="file size conflicts"):
        p0._safe_tree_entries(
            [
                {
                    "type": "file",
                    "path": "checkpoints/last.safetensors",
                    "size": 4095,
                    "lfs": {"oid": "2" * 64, "size": 4096},
                }
            ]
        )


def test_empty_or_malformed_safetensors_inventory_fails_closed():
    with pytest.raises(sync.IntegrityError, match="contains no tensors"):
        p0._validated_safetensors_summary(
            {"__metadata__": {}}, lfs_bytes=4096, raw_header_bytes=64
        )
    with pytest.raises(sync.IntegrityError, match="byte span contradicts"):
        p0._validated_safetensors_summary(
            {
                "layer.weight": {
                    "dtype": "F16",
                    "shape": [2],
                    "data_offsets": [0, 3],
                }
            },
            lfs_bytes=4096,
            raw_header_bytes=64,
        )
    with pytest.raises(sync.IntegrityError, match="not contiguous"):
        p0._validated_safetensors_summary(
            {
                "layer.weight": {
                    "dtype": "F16",
                    "shape": [2],
                    "data_offsets": [1, 5],
                }
            },
            lfs_bytes=69,
            raw_header_bytes=64,
        )
    with pytest.raises(sync.IntegrityError, match="do not cover"):
        p0._validated_safetensors_summary(
            {
                "layer.weight": {
                    "dtype": "F16",
                    "shape": [2],
                    "data_offsets": [0, 4],
                }
            },
            lfs_bytes=69,
            raw_header_bytes=64,
        )


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(sync.IntegrityError, match="not valid JSON"):
        p0._json_bytes(b'{"tensor":{},"tensor":{}}', "duplicate-key fixture")


def test_percent_encoded_checkpoint_suffix_cannot_evade_raw_body_guard():
    digest = "a" * 64
    assert p0._declared_raw_weight_objects(
        {"path": "checkpoints/model.%73afetensors", "object_sha256": digest}
    ) == {f"objects/sha256/{digest[:2]}/{digest}"}


def test_output_is_create_only(tmp_path):
    result, _package, inputs_and_output = run_build(tmp_path)
    api, watcher, headers, inventory, output = inputs_and_output
    package_path = output / result["package"]
    package_path.write_text("corrupt\n")
    before = package_path.read_bytes()
    with pytest.raises(sync.IntegrityError, match="different bytes"):
        p0.build_package(
            api, watcher, headers, inventory, output, TOURNAMENT, observed_at=NOW
        )
    assert package_path.read_bytes() == before


def test_watcher_fixture_bodies_are_never_reclassified_as_training_archives():
    rows, missing = p0.public_fixture_inventory([], {}, {TASK})
    assert rows[0]["actual_train_zip_image_count"] is None
    assert missing == [
        {"class": "public-training-archive-counts-unavailable", "task_id": TASK}
    ]


def test_complete_training_archive_integration_exposes_actual_and_dedup_counts(
    tmp_path, monkeypatch
):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_url = "https://s3.eu-central-003.backblazeb2.com/training.zip"
    output_buffer = io.BytesIO()
    with zipfile.ZipFile(output_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate((b"first", b"second", b"first")):
            archive.writestr(f"dataset/{index:03d}.png", image)
            archive.writestr(f"dataset/{index:03d}.txt", f"caption {index}")
    archive_body = output_buffer.getvalue()
    task_body = json.dumps(
        {"task_id": TASK, "status": "completed", "training_data": training_url}
    ).encode()
    requested: list[str] = []

    class Response:
        def __init__(self, body: bytes, headers: dict[str, str]):
            self.status = 200
            self.headers = headers
            self.body = body
            self.offset = 0

        def read(self, amount=-1):
            if amount < 0:
                amount = len(self.body) - self.offset
            start = self.offset
            self.offset = min(len(self.body), self.offset + amount)
            return self.body[start : self.offset]

        def close(self):
            pass

    def opener(request, timeout):
        del timeout
        requested.append(request.full_url)
        if request.full_url == f"https://api.gradients.io/auditing/tasks/{TASK}":
            return Response(task_body, {"Content-Type": "application/json"})
        if request.full_url == training_url:
            return Response(
                archive_body,
                {
                    "Content-Type": "application/zip",
                    "Content-Length": str(len(archive_body)),
                },
            )
        raise AssertionError(request.full_url)

    training_root = tmp_path / "training-archives"
    capture_result = capture.capture(
        api,
        training_root,
        TOURNAMENT,
        observed_at=NOW,
        opener=opener,
    )
    training_inventory = training_root / capture_result["inventory"]
    checksum = training_root / capture_result["checksum"]
    captured_inventory = json.loads(training_inventory.read_text())
    receipt = json.loads(
        (training_root / captured_inventory["tasks"][0]["receipt"]).read_text()
    )
    archive_path = training_root / receipt["archive"]["object"]
    archive_relative = archive_path.relative_to(training_root).as_posix()
    original_read_regular = p0._read_regular

    def reject_buffered_archive_read(root, relative, limit=p0.MAX_JSON_BYTES):
        if Path(root) == training_root.absolute() and relative == archive_relative:
            raise AssertionError("a COMPLETE ZIP must be hashed as a stream")
        return original_read_regular(root, relative, limit)

    monkeypatch.setattr(p0, "_read_regular", reject_buffered_archive_read)
    output = tmp_path / "output"
    result = p0.build_package(
        api,
        watcher,
        headers,
        header_inventory,
        output,
        TOURNAMENT,
        observed_at=NOW,
        training_archive_root=training_root,
        training_archive_inventory=training_inventory,
        training_archive_checksum=checksum,
    )

    assert result["status"] == "COMPLETE"
    assert requested == [
        f"https://api.gradients.io/auditing/tasks/{TASK}",
        training_url,
    ]
    package = json.loads((output / result["package"]).read_text())
    row = package["public_training_archive_inventory"][0]
    assert row["status"] == "COMPLETE"
    assert row["counts_available"] is True
    assert row["actual_train_zip_image_count"] == 3
    assert row["post_byte_dedup_image_count"] == 2
    assert row["rights"]["fixture_admission"] == "not admitted"
    provenance = package["inputs"]["public_training_archives"]
    assert provenance["status"] == "COMPLETE"
    assert provenance["verified_complete_archive_count"] == 1
    assert provenance["verified_complete_archive_bytes"] == archive_path.stat().st_size
    assert not package["completeness"]["missing_or_partial_surfaces"]


def test_partial_training_receipt_remains_partial_with_unavailable_counts(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(
        tmp_path, api, partial=True
    )
    output = tmp_path / "output"
    result = p0.build_package(
        api,
        watcher,
        headers,
        header_inventory,
        output,
        TOURNAMENT,
        observed_at=NOW,
        training_archive_root=training_root,
        training_archive_inventory=training_inventory,
    )

    assert result["status"] == "PARTIAL"
    package = json.loads((output / result["package"]).read_text())
    row = package["public_training_archive_inventory"][0]
    assert row["status"] == "PARTIAL"
    assert row["counts_available"] is False
    assert row["actual_train_zip_image_count"] is None
    assert row["post_byte_dedup_image_count"] is None
    assert package["inputs"]["public_training_archives"]["status"] == "PARTIAL"
    assert any(
        item["class"] == "public-training-archive-counts-unavailable"
        for item in package["completeness"]["missing_or_partial_surfaces"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("schema", "wrong.schema"), ("schema_version", 2), ("schema_version", True)],
)
def test_training_inventory_schema_and_version_fail_closed(
    tmp_path, field, replacement
):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    value = json.loads(training_inventory.read_text())
    value[field] = replacement
    rewrite_training_inventory(training_inventory, value)

    with pytest.raises(sync.IntegrityError, match="schema/version"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_inventory_checksum_mismatch_fails_closed(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, checksum, _ = training_archive_inputs(tmp_path, api)
    checksum.write_text(f"{'0' * 64}  {training_inventory.name}\n", encoding="ascii")

    with pytest.raises(sync.IntegrityError, match="checksum sidecar"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_inventory_must_have_exactly_one_row_for_each_r1_task(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    value = json.loads(training_inventory.read_text())
    value["tasks"].append(dict(value["tasks"][0]))
    value["task_count"] = 2
    value["complete_task_count"] = 2
    rewrite_training_inventory(training_inventory, value)

    with pytest.raises(sync.IntegrityError, match="exactly one row per R1 task"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_inventory_task_identity_mismatch_fails_closed(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    value = json.loads(training_inventory.read_text())
    value["tasks"][0]["task_id"] = "22222222-2222-4222-8222-222222222222"
    rewrite_training_inventory(training_inventory, value)

    with pytest.raises(sync.IntegrityError, match="task identity mismatch"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_receipt_checksum_and_task_binding_are_verified(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    inventory_value = json.loads(training_inventory.read_text())
    receipt_path = training_root / inventory_value["tasks"][0]["receipt"]
    receipt_value = json.loads(receipt_path.read_text())
    receipt_value["task_id"] = "22222222-2222-4222-8222-222222222222"
    receipt_body = sync.canonical_json(receipt_value)
    receipt_path.write_bytes(receipt_body)
    inventory_value["tasks"][0]["receipt_sha256"] = sha(receipt_body)
    rewrite_training_inventory(training_inventory, inventory_value)

    with pytest.raises(sync.IntegrityError, match="receipt tournament/task/status identity"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_receipt_checksum_mismatch_fails_closed(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    inventory_value = json.loads(training_inventory.read_text())
    receipt_path = training_root / inventory_value["tasks"][0]["receipt"]
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(sync.IntegrityError, match="receipt checksum mismatch"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_api_snapshot_identity_mismatch_fails_closed(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    value = json.loads(training_inventory.read_text())
    value["input_safe_api_snapshot"]["sha256"] = "0" * 64
    rewrite_training_inventory(training_inventory, value)

    with pytest.raises(sync.IntegrityError, match="safe API snapshot identity mismatch"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_complete_training_archive_sha_and_size_are_verified(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, archive_path = training_archive_inputs(tmp_path, api)
    assert archive_path is not None
    archive_path.write_bytes(archive_path.read_bytes() + b"tamper")

    with pytest.raises(sync.IntegrityError, match="archive size mismatch"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_complete_training_archive_same_size_sha_mismatch_fails_closed(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, archive_path = training_archive_inputs(tmp_path, api)
    assert archive_path is not None
    body = bytearray(archive_path.read_bytes())
    body[0] ^= 0xFF
    archive_path.write_bytes(body)

    with pytest.raises(sync.IntegrityError, match="archive SHA/size mismatch"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/training.zip",
        "https://s3.eu-central-003.backblazeb2.com:444/training.zip",
        "https://s3.eu-central-003.backblazeb2.com.evil.example/training.zip",
    ],
)
def test_training_receipt_transfer_authority_is_revalidated_at_consumption(
    tmp_path, url
):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)
    rewrite_training_receipt(
        training_root,
        training_inventory,
        lambda receipt: receipt["source_training_archive"].__setitem__("url", url),
    )

    with pytest.raises(sync.IntegrityError, match="transfer provenance violates"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_receipt_cannot_relabel_caption_as_image(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)

    def relabel(receipt):
        caption = next(
            row for row in receipt["inventory"]["members"] if row["path"].endswith(".txt")
        )
        caption["kind"] = "image"

    rewrite_training_receipt(training_root, training_inventory, relabel)

    with pytest.raises(sync.IntegrityError, match="contradicts the verified ZIP bytes"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_receipt_member_hashes_and_dedup_count_are_rederived_from_zip(tmp_path):
    api, watcher, headers, header_inventory = inputs(tmp_path)
    training_root, training_inventory, _, _ = training_archive_inputs(tmp_path, api)

    def forge_inventory(receipt):
        third_image = [
            row for row in receipt["inventory"]["members"] if row["kind"] == "image"
        ][2]
        third_image["sha256"] = "a" * 64
        receipt["inventory"]["post_exact_byte_dedup_image_count"] = 3

    rewrite_training_receipt(training_root, training_inventory, forge_inventory)
    inventory = json.loads(training_inventory.read_bytes())
    inventory["tasks"][0]["post_exact_byte_dedup_image_count"] = 3
    rewrite_training_inventory(training_inventory, inventory)

    with pytest.raises(sync.IntegrityError, match="contradicts the verified ZIP bytes"):
        p0.build_package(
            api,
            watcher,
            headers,
            header_inventory,
            tmp_path / "output",
            TOURNAMENT,
            observed_at=NOW,
            training_archive_root=training_root,
            training_archive_inventory=training_inventory,
        )


def test_training_archive_cli_arguments_are_exposed():
    args = p0.parse_args(
        [
            "--api-snapshot",
            "/api.json",
            "--watcher-root",
            "/watcher",
            "--header-root",
            "/headers",
            "--header-inventory",
            "/headers/inventory.json",
            "--training-archive-root",
            "/training",
            "--training-archive-inventory",
            "/training/inventory.json",
            "--training-archive-checksum",
            "/training/inventory.sha256",
            "--output-root",
            "/output",
            "--tournament-id",
            TOURNAMENT,
        ]
    )
    assert args.training_archive_root == Path("/training")
    assert args.training_archive_inventory == Path("/training/inventory.json")
    assert args.training_archive_checksum == Path("/training/inventory.sha256")
