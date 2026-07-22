"""Build the ai-toolkit config YAML from a template + the task spec.

Loads the bundled per-type template, overrides ONLY the contract keys
(name==repo, paths, trigger, steps/save), and injects the text-encoder / VAE
paths per type (exactly as the god_ref entrypoint does). Never raises out:
``build_config`` degrades to the raw template with just the load-bearing
name/paths patched so an override bug can't forfeit the task (INV-1). The single
non-negotiable is ``config.name == expected_repo_name`` — otherwise the validator
uploader sees an empty folder ("Nothing to upload").
"""

from __future__ import annotations

import os

import yaml

from forge import recipe

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
    cfg = load_template(spec.model_type)  # may raise → caller wraps
    try:
        return _apply_overrides(cfg, spec, num_images, hours_to_complete)
    except Exception:
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
        return cfg


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
        # GATE-B PROVISIONAL LR (Jul-21/22 H100 ladder; single public fixture,
        # single training seed — REVERSIBLE, revisit with a second fixture):
        # 8 arms x 5 candidates, all 40 rescored by the pinned exact evaluator.
        # Best condition: 367 steps @ 2e-4, guidance on, final export (0.02711);
        # the restored 1e-4 baseline arm scored 0.02842 (-4.63%), all 4 external
        # holdout images improved. Note the LRxGUIDANCE INTERACTION: guidance-on
        # wins at 2e-4 but guidance-off edged it at 1e-4/5e-5 — this pins the
        # best measured PAIR, not a universal guidance rule. Evidence record:
        # project docs SN56-GATE-B-H100-RESULTS.md §5 (kept outside this public
        # repo deliberately — counter-intel).
        p["train"]["lr"] = 2e-4
    return cfg


def write_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
