"""Build the ai-toolkit config YAML from a template + the task spec.

Loads the bundled per-type template, overrides ONLY the contract keys
(name==repo, paths, trigger, steps/save), and injects the text-encoder / VAE
paths per type (exactly as the god_ref entrypoint does). On the normal production
path, ``build_config`` degrades to the raw template with just the load-bearing
name/paths patched so an override bug can't forfeit the task (INV-1). An explicit
but invalid calibration profile is the sole fail-closed exception. The single
non-negotiable is ``config.name == expected_repo_name`` — otherwise the validator
uploader sees an empty folder ("Nothing to upload").
"""

from __future__ import annotations

import hashlib
import json
import os

import yaml

from forge import krea_calibration_profiles, recipe

# Templates are shipped INSIDE the package (forge/templates/*.yaml) so they are
# present under any deployment (source COPY, `pip install .` wheel, or local test)
# — the old repo-root ../templates path was invisible to setuptools and would
# silently forfeit EVERY task if this repo were ever wheel-installed. We still
# honour FORGE_TEMPLATES_DIR (Docker) first, then fall back through the packaged
# dir and the legacy repo-root dir, so a missing/relocated dir can't forfeit.
_PKG_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "templates"
)
_REPO_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
)
_TEMPLATES_DIR = os.environ.get("FORGE_TEMPLATES_DIR", _PKG_TEMPLATES_DIR)
_TEMPLATE_BY_TYPE = {
    "flux": "base_diffusion_flux.yaml",
    "krea2": "base_diffusion_krea2.yaml",
    "ideogram4": "base_diffusion_ideogram4.yaml",
    # z-image / qwen-image: templates are fully self-contained (assistant-LoRA
    # and uint3-adapter paths baked in, staged by the validator's downloader) —
    # no per-type injection needed beyond the standard overrides.
    "z-image": "base_diffusion_zimage.yaml",
    "qwen-image": "base_diffusion_qwen_image.yaml",
}
_IDEOGRAM4_TE = "/cache/hf_cache/Qwen--Qwen3-VL-8B-Instruct"
_KREA2_TE = "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct"


def _template_dirs() -> list[str]:
    """Ordered, de-duplicated search path: env/primary → packaged → repo-root."""
    out: list[str] = []
    for d in (_TEMPLATES_DIR, _PKG_TEMPLATES_DIR, _REPO_TEMPLATES_DIR):
        if d and d not in out:
            out.append(d)
    return out


def load_template(model_type: str) -> dict:
    fname = _TEMPLATE_BY_TYPE[model_type]
    for d in _template_dirs():
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            with open(path) as fh:
                return yaml.safe_load(fh)
    # None found: raise a clear error against the primary dir (caller degrades).
    with open(os.path.join(_TEMPLATES_DIR, fname)) as fh:
        return yaml.safe_load(fh)


def resolve_base_model(cached_model_dir: str) -> str:
    """Return the cache DIR the ai-toolkit loader wants — never collapsed to a file.

    Matches the AUTHORITATIVE runtime entrypoint (image_toolkit_entrypoint.py):
    ``model_config['name_or_path'] = str(model_path)`` UNCONDITIONALLY, so the
    ai-toolkit loader can resolve per-arch subfolders (vae/, text_encoder, …).
    (Only the LEGACY training_paths helper collapsed a single .safetensors to the
    file; the entrypoint that actually runs does not, and collapsing to a bare
    file breaks subfolder resolution → base-model-load / zero-score risk.)
    """
    return cached_model_dir


def build_config(spec, num_images, hours_to_complete) -> dict:
    # The selector is deliberately resolved before the never-forfeit override
    # wrapper: an explicit but invalid calibration request must fail closed and
    # must never silently fall through to the production recipe.
    calibration_profile = krea_calibration_profiles.selected_profile(spec.model_type)
    calibration_depth = krea_calibration_profiles.selected_stage2_depth(
        spec.model_type, calibration_profile
    )
    stage2_control = krea_calibration_profiles.selected_stage2_run_control(
        spec.model_type, calibration_profile
    )
    if (
        stage2_control is not None
        and calibration_profile is not None
        and calibration_profile.profile_id != "K0"
        and calibration_depth is None
    ):
        raise krea_calibration_profiles.KreaCalibrationProfileError(
            "Stage-2 non-control run controls require measured depth binding"
        )
    cfg = load_template(spec.model_type)  # may raise → caller wraps
    try:
        resolved = _apply_overrides(cfg, spec, num_images, hours_to_complete)
    except Exception:
        if calibration_profile is not None:
            # Experiment/Stage-2 evidence must never be generated from the
            # never-forfeit degraded config while carrying a frozen arm label.
            raise
        # Degrade to the template with only the load-bearing name/paths patched so
        # an override bug can't forfeit (INV-1). name==repo is non-negotiable.
        try:
            cfg["config"]["name"] = spec.expected_repo_name
            p = cfg["config"]["process"][0]
            p["training_folder"] = spec.training_folder
            p["datasets"][0]["folder_path"] = spec.dataset_images_dir
            p["model"]["name_or_path"] = resolve_base_model(spec.cached_model_dir)
        except Exception:
            pass
        # TE/vae injection separately: on the validator's airgapped box the
        # loaders' HF-id defaults can't download, so losing this patch would
        # turn a degraded-but-trainable run into a crash.
        try:
            p = cfg["config"]["process"][0]
            mk = p["model"].setdefault("model_kwargs", {})
            if spec.model_type == "ideogram4":
                mk["text_encoder_path"] = _IDEOGRAM4_TE
            elif spec.model_type == "krea2":
                mk["text_encoder_path"] = _KREA2_TE
                mk["vae_path"] = spec.cached_model_dir
        except Exception:
            pass
        # Apply the same fixed candidate/I/O budget even on the degraded path;
        # the raw template's 200-250 cadence would miss most short jobs.
        try:
            p = cfg["config"]["process"][0]
            p["save"]["save_every"] = recipe.kill_safe_save_every(
                p["train"]["steps"], p["save"].get("save_every", 250)
            )
        except Exception:
            pass
        resolved = cfg

    # Environment-unset production returns the same object produced above with
    # no extra metadata, event, or recipe mutation.  The calibration path stays
    # dormant until the selector is explicitly present.
    if calibration_profile is not None:
        resolved = krea_calibration_profiles.apply_profile(
            resolved, calibration_profile, depth_override=calibration_depth
        )
    return krea_calibration_profiles.apply_stage2_run_control(resolved, stage2_control)


def _apply_overrides(cfg, spec, num_images, hours_to_complete) -> dict:
    cfg["config"]["name"] = spec.expected_repo_name  # MUST == repo_name
    p = cfg["config"]["process"][0]  # process is a LIST
    p["training_folder"] = spec.training_folder
    p["trigger_word"] = spec.trigger_word  # None → null (flux has no key; set it)
    p["datasets"][0]["folder_path"] = spec.dataset_images_dir

    model = p.setdefault("model", {})
    model["name_or_path"] = resolve_base_model(spec.cached_model_dir)

    template_steps = p["train"]["steps"]
    steps = recipe.size_scaled_steps(
        spec.model_type, num_images, hours_to_complete, template_steps
    )
    p["train"]["steps"] = steps
    p["save"]["save_every"] = recipe.kill_safe_save_every(
        steps, p["save"].get("save_every", 250)
    )

    if spec.model_type == "ideogram4":
        mk = model.setdefault("model_kwargs", {})
        mk["text_encoder_path"] = _IDEOGRAM4_TE
        # unconditional_lora_path already in the template model block — PRESERVED.
        # Calibrated Jul 16 on real photos with a true holdout, scored by the
        # validator's own eval stack: the template lr 4e-4 makes adjacent
        # checkpoints swing 2x in score (0.030<->0.075), while 1e-4 tracks a
        # stable curve that BEATS the base model (best 0.0290 vs zero-LoRA
        # 0.0351). Predictability matters: we cannot checkpoint-pick in-tourney.
        p["train"]["lr"] = 1e-4
    elif spec.model_type == "krea2":
        mk = model.setdefault("model_kwargs", {})
        mk["text_encoder_path"] = _KREA2_TE
        # Krea2Model appends the "vae" subfolder itself → pass the model DIR.
        mk["vae_path"] = spec.cached_model_dir
        # LR OVERRIDE REMOVED (Jul-20 postmortem): the 1e-3 override came from
        # a 128-step / 2-holdout-image probe and failed at tournament scale —
        # our R2 krea2 (lr 1e-3, 367 steps, final-export) scored 0.1420 vs the
        # opponent's 0.0525 on template lr 1e-4 with a deep run + EARLY
        # selected checkpoint. Template LR stands; the real gap is checkpoint
        # SELECTION for image exports (see postmortem handoff).
    return cfg


def write_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rendered = yaml.safe_dump(cfg, sort_keys=False)
    with open(path, "w") as fh:
        fh.write(rendered)
    _write_stage2_control_receipt(cfg, rendered.encode("utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_stage2_control_receipt(cfg: dict, config_bytes: bytes) -> None:
    """Emit the private, create-only proof that Stage-2 controls were consumed."""

    raw_profile = os.environ.get(krea_calibration_profiles.PROFILE_SELECTOR_ENV)
    if raw_profile is None and all(
        os.environ.get(name) is None
        for name in (
            krea_calibration_profiles.STAGE2_SEED_ENV,
            krea_calibration_profiles.STAGE2_PLAN_SHA_ENV,
            krea_calibration_profiles.STAGE2_RECEIPT_PATH_ENV,
            krea_calibration_profiles.STAGE2_TARGET_NUMERATOR_ENV,
            krea_calibration_profiles.STAGE2_TARGET_DENOMINATOR_ENV,
        )
    ):
        return
    profile = krea_calibration_profiles.profile_for_id(raw_profile or "")
    control = krea_calibration_profiles.selected_stage2_run_control("krea2", profile)
    if (
        control is None
    ):  # pragma: no cover - explicit variables above make this impossible.
        raise krea_calibration_profiles.KreaCalibrationProfileError(
            "Stage-2 receipt lacks a validated run control"
        )
    process = cfg["config"]["process"][0]
    train = process["train"]
    network = process["network"]
    dataset = process["datasets"][0]
    save = process["save"]
    model = process["model"]
    checkpoint_selection = cfg.get("meta", {}).get("forge_krea_checkpoint_selection")
    if not isinstance(checkpoint_selection, dict):
        raise krea_calibration_profiles.KreaCalibrationProfileError(
            "Stage-2 config lacks its checkpoint-selection binding"
        )
    raw_steps = os.environ.get(krea_calibration_profiles.STAGE2_STEPS_ENV)
    raw_throughput = os.environ.get(krea_calibration_profiles.STAGE2_THROUGHPUT_SHA_ENV)
    if process.get("training_seed") != control.seed:
        raise krea_calibration_profiles.KreaCalibrationProfileError(
            "Stage-2 process did not consume its training seed"
        )
    if profile.profile_id == "K0":
        if raw_steps is not None or raw_throughput is not None:
            raise krea_calibration_profiles.KreaCalibrationProfileError(
                "K0 receipt cannot carry a Stage-2 depth override"
            )
        throughput_sha = None
    else:
        if raw_steps != str(train.get("steps")) or raw_throughput is None:
            raise krea_calibration_profiles.KreaCalibrationProfileError(
                "Stage-2 receipt depth differs from the effective config"
            )
        throughput_sha = raw_throughput
    effective = {
        "config_name": cfg["config"].get("name"),
        "training_folder": process.get("training_folder"),
        "trigger_word": process.get("trigger_word"),
        "model_arch": model.get("arch"),
        "model_name_or_path": model.get("name_or_path"),
        "model_kwargs": model.get("model_kwargs"),
        "dataset_folder_path": dataset.get("folder_path"),
        "network_rank": network.get("linear"),
        "network_alpha": network.get("linear_alpha"),
        "optimizer": train.get("optimizer"),
        "optimizer_params": train.get("optimizer_params"),
        "loss": train.get("loss_type"),
        "guidance_enabled": train.get("do_differential_guidance"),
        "guidance_scale": train.get("differential_guidance_scale"),
        "learning_rate": train.get("lr"),
        "dropout": dataset.get("caption_dropout_rate"),
        "ema": train.get("ema_config"),
        "steps": train.get("steps"),
        "save_every": save.get("save_every"),
        "push_to_hub": save.get("push_to_hub"),
        "batch_size": train.get("batch_size"),
        "gradient_accumulation": train.get("gradient_accumulation"),
        "resolution": dataset.get("resolution"),
        "train_dtype": train.get("dtype"),
        "save_dtype": save.get("dtype"),
        "cache_latents_to_disk": dataset.get("cache_latents_to_disk"),
        "cache_text_embeddings": train.get("cache_text_embeddings"),
        "compile": model.get("compile"),
        "dataloader_workers": dataset.get("num_workers", 0),
    }
    body = {
        "schema": 1,
        "kind": "forge-krea-stage2-config-control-receipt",
        "execution_plan_sha256": control.execution_plan_sha256,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "training_seed": control.seed,
        "throughput_profile_sha256": throughput_sha,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "effective_config_file": {
            "path": "effective-config.yaml",
            "bytes": len(config_bytes),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "effective_recipe": effective,
        "effective_recipe_sha256": hashlib.sha256(
            _canonical_bytes(effective)
        ).hexdigest(),
        "checkpoint_selection": checkpoint_selection,
        "release_authorized": False,
    }
    receipt = {
        **body,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }
    payload = _canonical_bytes(receipt) + b"\n"
    parent = os.path.dirname(control.receipt_path)
    if os.path.realpath(parent) != parent or not os.path.isdir(parent):
        raise krea_calibration_profiles.KreaCalibrationProfileError(
            "Stage-2 receipt parent is not the precreated real directory"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    config_copy_path = os.path.join(parent, "effective-config.yaml")
    config_fd = os.open(config_copy_path, flags, 0o444)
    try:
        with os.fdopen(config_fd, "wb", closefd=False) as handle:
            handle.write(config_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(config_fd)
    fd = os.open(control.receipt_path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
