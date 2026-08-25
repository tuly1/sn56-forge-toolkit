"""Validated Ideogram product-category promotion layered on production 59e."""

from __future__ import annotations

from collections import Counter
import copy
import glob
import hashlib
import json
import math
import os
import re
from typing import Any, Mapping

from forge import ideogram_release_policy, recipe, telemetry


POLICY_ID = "week11-ideogram-product-content-exact-final-v1"
POLICY_KIND = "forge-ideogram-week11-product-content-policy"
ACTIVATION_KIND = "forge-ideogram-week11-product-content-activation"
CHECKPOINT_MAPPING_RULE = "nearest_current_candidate_ties_choose_earlier_step"
BASE_PROJECTION_SHA256 = (
    "cbc9e2689fee11d8842bd2a90651439eba6232e80db89ac8a16199adb505f5fd"
)
PRODUCT_PROJECTION_SHA256 = (
    "152d0c893293bd1eb5ddce80be1a50e2c65d0030c3e8db994421e10fb39c6733"
)
FROZEN_ONE_HOUR_PROJECTION_SHA256 = (
    "ce226348ca641932de4a0f27d59a69ad8a124825068a4f1e9e9d134166eba4cd"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MIN_CATEGORY_FRACTION = 0.25
_CONTENT_SECONDS_PER_STEP = 2.1
_UPLOAD_RESERVE_S = 420
_MIN_STEPS = 120
_MAX_STEPS = 1700

# Exact insertion order and keywords from public source 0ab94013... .
_CATEGORY_KEYWORDS = {
    "logo": ("logo", "brandmark", "wordmark", "monogram", "emblem", "brand mark"),
    "line": ("line art", "line-art", "lineart", "linework", "schematic", "blueprint"),
    "watercolor": ("watercolor", "gouache", "aquarelle"),
    "social": ("infographic", "headline", "banner", "social"),
    "design": ("ui", "interface", "mobile", "dashboard", "mockup", "light mode"),
    "product": ("product", "packshot", "studio shot", "macro", "lifestyle"),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


SOURCE_BINDINGS: Mapping[str, Any] = {
    "decision_sha256": "aa9c66a59cf51b3338330cac9ceb6ccaa4b84467f223d1323d17c6de502a715a",
    "sync_proof_sha256": "4c9581cd1ae909f58e46ff6b0ca87bfcb2a010481162bf6281db07595f742b75",
    "scientific_core_sha256": "62f8fbd6c10b76ab69ea806462a3f593da70fc38e2a1d6ecd8aa6adf42173223",
    "confirmation_freeze_sha256": "41f3adeb69dac804e559f5b38e8b4a147ba2e0e93f8d4958151f699bd959cb33",
    "runtime_canary_sha256": "9eebb82d26e8afc6534b63e2310b20e5574fb1f64940be43be8268011eb5d1e4",
    "scores": {
        "seed42_control": "234b3ced6f6a37109791856acbd8fa09588df8e05bb9f4265486d79bc1dba6a6",
        "seed42_challenger": "21cfe7ac30ac8566357cb4af2bce9c91cb3c0cc60f0ec56ee09f9e285071fc1d",
        "seed43_control": "2b96d6452bc681119581f147ddb9af64c00fa0a624efbe984bacb7caa886a76d",
        "seed43_challenger": "34459e42894eadeb2bbc97e5bf14d5a5a9712c04159d636cd689d4bc48a6a427",
    },
    "artifacts": {
        "control_final": "bae616576d394f26a75ada6726d0abbc48ea0bbba389559b814279e23e43d275",
        "challenger_final": "2748321ab8b93d82c208d398525722b7f87d011954751a66ccd879cdfdeb6f1d",
        "fixture": "7a5f3d460a5ce932caa806d2c364aeadfafa43d8c2631243e0072568a540cafe",
        "scorer": "3458e451d72ef81d9f8c2ce85241b26916a2565ba40a08adfcf404b929499e05",
        "workflow": "88e7de1bce05ec96f5527fdafa2a53fdd8e3b0a8daf59af853a9e48d98a64143",
        "challenger_config_file": "86a2bdbd3cb431040e288a393dd96b92c8dbbef3aa25f57ad3b12de22c6ea3bd",
    },
    "runtime": {
        "ai_toolkit_commit": "99be3d96a2468d3a5228a4eb05ba67e63c586b4e",
        "god_commit": "f7caab6cb2786f4210036b0675123d6ba9633e8f",
        "god_tree": "e8f1b855cc9468abfc91743a4e42a193238e97df",
        "comfyui_commit": "091b70edda0c062fc9338a1d7e8e2f94f4c0ad0b",
        "comfyui_tree": "1936f65713a6a6d88066b0d6127931ec50c1a2c1",
        "tooling_nodes_commit": "5d3194f4d4158ab31df7a060e1e4c56fa03f320c",
        "tooling_nodes_tree": "c7f2378076420703e933bb7619f5f1d67eb1dbeb",
        "python": "3.10.12",
        "torch": "2.9.1+cu128",
        "distributions_sha256": "08141a902291003765840930a39e0ce9657dd63d4f7a11cf2c21b3bb0ca1be85",
        "identity_marker_sha256": "db6c8a96f25493eda9f74f23f0b5f248a8b50a5b469b15c5ee7313875b416364",
    },
    "public_source": {
        "commit": "57b0bcd07bbddffd8774bae20ac0f33a7f0ee509",
        "tree": "8210f96bf6af54a13d562efdf0fc6abd13246556",
        "detector_source_sha256": "0ab94013defe308280f4bb6b6f925c2748d6f58ef66d71feccadb6f35e01b3b5",
        "template_sha256": "86c56bbbe210317782796937f5a67c3f98cd5444c63bbff7caac0fc2d3955b5d",
        "dockerfile_sha256": "bf7c79120ba5aec77bc31aec717ab4f16e2627b01a5152b8a3a968bc6c44c62a",
    },
    "production": {
        "commit": "59e0698c952edaf1bf34a117ecad41bce87517cf",
        "tree": "613a1cc2d750731df007cc9b2b49e461d0ae368f",
        "policy_source_sha256": "f59a178bdba720faf60a650187994cce370c0be80a01576d287f64b827b0bcbb",
        "recipe_source_sha256": "ff6ac2b813961eb894e7de5b3fe2e7ba5373f9ffa7767135750458f58b98eede",
        "template_sha256": "c8344794600c78f6fe4f3acd897233906d6289513ce26cdf9fc966ea54f2e438",
        "runtime_lock_sha256": "9c4c15130508c547c67d891f559ca1a513cd62bd5a4b695eb25ceafccd0b850b",
    },
}
SOURCE_BINDINGS_SHA256 = (
    "fcba868d3527993dc160d1d9500668602efcb445f961e2ac19b30cc90e5612c5"
)

_POLICY_BODY = {
    "schema": 1,
    "kind": POLICY_KIND,
    "policy_id": POLICY_ID,
    "activation_category": "product",
    "source_bindings_sha256": SOURCE_BINDINGS_SHA256,
    "base_projection_sha256": BASE_PROJECTION_SHA256,
    "product_projection_sha256": PRODUCT_PROJECTION_SHA256,
    "frozen_one_hour_projection_sha256": FROZEN_ONE_HOUR_PROJECTION_SHA256,
    "step_policy": {
        "seconds_per_step": _CONTENT_SECONDS_PER_STEP,
        "upload_reserve_s": _UPLOAD_RESERVE_S,
        "minimum": _MIN_STEPS,
        "maximum": _MAX_STEPS,
    },
    "checkpoint_policy": {
        "target_fraction": {"numerator": 1, "denominator": 1},
        "mapping_rule": CHECKPOINT_MAPPING_RULE,
        "calibration_artifact": "unnumbered_exact_final",
    },
    "release_authorized": True,
    "deployment_authorized": False,
}
POLICY_SHA256 = _sha256(_POLICY_BODY)

PRODUCTION_ACTIVATION: Mapping[str, Any] | None = {
    "schema": 1,
    "kind": ACTIVATION_KIND,
    "policy_sha256": "9e3b1e68d20c52932a3b1284267882970b11b6f5afa083d5763606de17951c96",
    "source_bindings_sha256": (
        "fcba868d3527993dc160d1d9500668602efcb445f961e2ac19b30cc90e5612c5"
    ),
    "selected_category": "product",
    "verdict": "PROMOTE",
    "verdict_scope": "CONTENT branch only",
    "production_mutation_authorized": True,
    "release_authorized": True,
    "deployment_authorized": False,
    "activation_sha256": "bfa59fc67196f9fba36644b48ab26a6583f5b348d2ef41b8fc20902c810ce918",
}


def detect_category(train_data_dir: str, trigger_word: str | None = None) -> str:
    """Return the exact public keyword-argmax category for recursive captions."""

    captions: list[str] = []
    try:
        if train_data_dir and os.path.isdir(train_data_dir):
            for path in glob.glob(
                os.path.join(train_data_dir, "**", "*.txt"), recursive=True
            ):
                try:
                    with open(path, encoding="utf-8", errors="ignore") as handle:
                        captions.append(handle.read().lower())
                except OSError:
                    pass
    except OSError:
        pass
    if not captions:
        return "default"
    counts: Counter[str] = Counter()
    for caption in captions:
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if re.search(
                    r"(?<![a-z])" + re.escape(keyword) + r"(?![a-z])", caption
                ):
                    counts[category] += 1
                    break
    if not counts:
        return "default"
    category, count = counts.most_common(1)[0]
    if count >= _MIN_CATEGORY_FRACTION * len(captions):
        return category
    return "default"


def product_steps(hours_to_complete: float) -> int:
    if (
        isinstance(hours_to_complete, bool)
        or not isinstance(hours_to_complete, (int, float))
        or not math.isfinite(float(hours_to_complete))
    ):
        raise ValueError("hours_to_complete must be finite")
    return max(
        _MIN_STEPS,
        min(
            _MAX_STEPS,
            int((float(hours_to_complete) * 3600 - _UPLOAD_RESERVE_S) / _CONTENT_SECONDS_PER_STEP),
        ),
    )


def _validated_sources() -> bool:
    try:
        if _sha256(SOURCE_BINDINGS) != SOURCE_BINDINGS_SHA256:
            raise ValueError("source binding hash differs")
        return True
    except Exception as exc:
        telemetry.event(
            "ideogram_content_policy_inactive",
            reason="invalid_source_bindings",
            error_type=type(exc).__name__,
        )
        return False


def _validated_activation(value: Any) -> dict[str, Any] | None:
    try:
        if not isinstance(value, Mapping):
            raise ValueError("activation is not a mapping")
        record = dict(value)
        expected_keys = {
            "schema",
            "kind",
            "policy_sha256",
            "source_bindings_sha256",
            "selected_category",
            "verdict",
            "verdict_scope",
            "production_mutation_authorized",
            "release_authorized",
            "deployment_authorized",
            "activation_sha256",
        }
        if set(record) != expected_keys:
            raise ValueError("activation keys differ")
        body = {k: v for k, v in record.items() if k != "activation_sha256"}
        if (
            record["schema"] != 1
            or record["kind"] != ACTIVATION_KIND
            or record["policy_sha256"] != POLICY_SHA256
            or record["source_bindings_sha256"] != SOURCE_BINDINGS_SHA256
            or record["selected_category"] != "product"
            or record["verdict"] != "PROMOTE"
            or record["verdict_scope"] != "CONTENT branch only"
            or record["production_mutation_authorized"] is not True
            or record["release_authorized"] is not True
            or record["deployment_authorized"] is not False
            or record["activation_sha256"] != _sha256(body)
        ):
            raise ValueError("activation identity or authority differs")
        return record
    except Exception as exc:
        telemetry.event(
            "ideogram_content_policy_inactive",
            reason="invalid_activation_record",
            error_type=type(exc).__name__,
        )
        return None


def _trainer_projection(cfg: Mapping[str, Any], plan_sentinel: str | None) -> dict[str, Any]:
    projection = copy.deepcopy(dict(cfg))
    meta = projection.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("config meta is invalid")
    projection["meta"] = {
        key: value for key, value in meta.items() if not key.startswith("forge_")
    }
    process = projection["config"]["process"][0]
    projection["config"]["name"] = "<runtime-name>"
    process["training_folder"] = "<runtime-path>"
    process["trigger_word"] = "<runtime-trigger>"
    process["datasets"][0]["folder_path"] = "<runtime-path>"
    process["model"]["name_or_path"] = "<runtime-path>"
    process["model"]["model_kwargs"]["text_encoder_path"] = "<runtime-path>"
    if plan_sentinel is not None:
        process["train"]["steps"] = f"<{plan_sentinel}-steps>"
        process["save"]["save_every"] = f"<{plan_sentinel}-save-every>"
    return projection


def _valid_base(
    cfg: Mapping[str, Any], num_images: int, hours_to_complete: float
) -> bool:
    try:
        if isinstance(num_images, bool) or not isinstance(num_images, int) or num_images <= 0:
            raise ValueError("num_images is invalid")
        if (
            isinstance(hours_to_complete, bool)
            or not isinstance(hours_to_complete, (int, float))
            or not math.isfinite(float(hours_to_complete))
        ):
            raise ValueError("hours_to_complete is invalid")
        process = cfg["config"]["process"][0]
        expected_steps = recipe.size_scaled_steps(
            "ideogram4", num_images, float(hours_to_complete), 2000
        )
        expected_save_every = recipe.kill_safe_save_every(
            expected_steps, 200, "ideogram4"
        )
        if (
            process["train"]["steps"] != expected_steps
            or process["save"]["save_every"] != expected_save_every
        ):
            raise ValueError("production plan differs")
        control = ideogram_release_policy.checkpoint_control(cfg)
        if control is None or control[1] != expected_steps:
            raise ValueError("production checkpoint binding differs")
        if _sha256(_trainer_projection(cfg, "production")) != BASE_PROJECTION_SHA256:
            raise ValueError("production projection differs")
        return True
    except Exception as exc:
        telemetry.event(
            "ideogram_content_policy_inactive",
            reason="invalid_production_base",
            error_type=type(exc).__name__,
        )
        return False


def apply(
    cfg: dict[str, Any],
    model_type: str,
    category: str | None,
    *,
    num_images: int,
    hours_to_complete: float,
    activation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically apply the validated product projection or return ``cfg``."""

    if model_type != "ideogram4" or category != "product":
        return cfg
    active = _validated_activation(
        PRODUCTION_ACTIVATION if activation is None else activation
    )
    if active is None or not _validated_sources():
        return cfg
    if not _valid_base(cfg, num_images, hours_to_complete):
        return cfg
    try:
        resolved = copy.deepcopy(cfg)
        process = resolved["config"]["process"][0]
        dataset = process["datasets"][0]
        network = process["network"]
        save = process["save"]
        train = process["train"]
        steps = product_steps(hours_to_complete)

        process.pop("training_seed", None)
        network.update({"linear": 32, "linear_alpha": 32})
        dataset.update(
            {"caption_dropout_rate": 0.05, "cache_latents_to_disk": True}
        )
        save.update({"save_every": 100, "max_step_saves_to_keep": 100})
        train.update(
            {
                "steps": steps,
                "lr": 0.0004,
                "unet_lr": 0.0004,
                "train_text_encoder": False,
                "cache_text_embeddings": True,
                "min_denoising_steps": 250,
                "timestep_type": "linear",
                "content_or_style": "balanced",
                "ema_config": {"use_ema": True, "ema_decay": 0.995},
            }
        )
        for key in (
            "text_encoder_lr",
            "lr_scheduler",
            "lr_scheduler_params",
            "do_cfg",
            "cfg_scale",
        ):
            train.pop(key, None)

        if _sha256(_trainer_projection(resolved, "product")) != PRODUCT_PROJECTION_SHA256:
            raise ValueError("product projection differs")
        if float(hours_to_complete) == 1.0 and (
            _sha256(_trainer_projection(resolved, None))
            != FROZEN_ONE_HOUR_PROJECTION_SHA256
        ):
            raise ValueError("one-hour challenger projection differs")

        resolved.setdefault("meta", {}).update(
            {
                "forge_ideogram_content_policy": {
                    "schema": 1,
                    "policy_id": POLICY_ID,
                    "policy_sha256": POLICY_SHA256,
                    "activation_sha256": active["activation_sha256"],
                    "source_bindings_sha256": SOURCE_BINDINGS_SHA256,
                    "base_projection_sha256": BASE_PROJECTION_SHA256,
                    "product_projection_sha256": PRODUCT_PROJECTION_SHA256,
                    "category": "product",
                    "hours_to_complete": float(hours_to_complete),
                    "release_authorized": True,
                    "deployment_authorized": False,
                },
                "forge_ideogram_content_checkpoint_selection": {
                    "schema": 1,
                    "mapping_rule": CHECKPOINT_MAPPING_RULE,
                    "target_fraction": {"numerator": 1, "denominator": 1},
                    "planned_steps": steps,
                    "selected_step": steps,
                    "calibration_artifact": "unnumbered_exact_final",
                },
            }
        )
        telemetry.event(
            "ideogram_content_policy_applied", category="product", planned_steps=steps
        )
        return resolved
    except Exception as exc:
        telemetry.event(
            "ideogram_content_policy_inactive",
            reason="application_failed",
            error_type=type(exc).__name__,
        )
        return cfg


def checkpoint_control(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], int] | None:
    """Return the product exact-final target, ahead of the generic control."""

    meta = cfg.get("meta")
    if not isinstance(meta, Mapping):
        return None
    binding = meta.get("forge_ideogram_content_policy")
    checkpoint = meta.get("forge_ideogram_content_checkpoint_selection")
    if binding is None and checkpoint is None:
        return None
    try:
        if not isinstance(binding, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError("partial content binding")
        active = _validated_activation(PRODUCTION_ACTIVATION)
        if active is None or not _validated_sources():
            raise ValueError("content authority is invalid")
        expected_binding = {
            "schema": 1,
            "policy_id": POLICY_ID,
            "policy_sha256": POLICY_SHA256,
            "activation_sha256": active["activation_sha256"],
            "source_bindings_sha256": SOURCE_BINDINGS_SHA256,
            "base_projection_sha256": BASE_PROJECTION_SHA256,
            "product_projection_sha256": PRODUCT_PROJECTION_SHA256,
            "category": "product",
            "hours_to_complete": binding.get("hours_to_complete"),
            "release_authorized": True,
            "deployment_authorized": False,
        }
        if dict(binding) != expected_binding:
            raise ValueError("content policy binding differs")
        hours = binding["hours_to_complete"]
        steps = product_steps(hours)
        expected_checkpoint = {
            "schema": 1,
            "mapping_rule": CHECKPOINT_MAPPING_RULE,
            "target_fraction": {"numerator": 1, "denominator": 1},
            "planned_steps": steps,
            "selected_step": steps,
            "calibration_artifact": "unnumbered_exact_final",
        }
        if dict(checkpoint) != expected_checkpoint:
            raise ValueError("content checkpoint binding differs")
        if cfg["config"]["process"][0]["train"]["steps"] != steps:
            raise ValueError("content steps differ")
        if _sha256(_trainer_projection(cfg, "product")) != PRODUCT_PROJECTION_SHA256:
            raise ValueError("content projection differs")
        return (
            {
                "fraction_numerator": 1,
                "fraction_denominator": 1,
                "selection_rule": CHECKPOINT_MAPPING_RULE,
            },
            steps,
        )
    except Exception as exc:
        raise ValueError("Ideogram content checkpoint binding is invalid") from exc
