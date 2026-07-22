"""Parse the validator's image-task arguments into a typed spec (ai-toolkit).

The image tournament dispatches on ``--model-type`` (flux | krea2 | ideogram4 |
z-image | qwen-image), not on a JSON dataset-type. The dataset arrives as a zip
of image+caption pairs pre-staged in the read-only cache; the ``--dataset-zip``
URL is a decoy (no internet at runtime). We resolve everything into one immutable
record here and expose the exact paths the ai-toolkit validator contract mandates.

vs the legacy SDXL/kohya schema: there is NO ``checkpoint/`` subfolder and NO
``instance_prompt``. ai-toolkit writes LoRA files FLAT into
``training_folder/{config.name}`` and injects the trigger via config
``trigger_word``, so ``output_dir == save_root`` and the DreamBooth prompt is gone.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

# Every current ai-toolkit image type routes through the same hardened runner.
KNOWN_MODEL_TYPES = ("flux", "krea2", "ideogram4", "z-image", "qwen-image")


@dataclass(frozen=True)
class ImageSpec:
    task_id: str
    model: str
    model_type: str
    expected_repo_name: str
    # A per-task trigger word. Optional: only sent when the task provides one.
    # ai-toolkit injects it via config `trigger_word` — we never prepend it to
    # captions ourselves.
    trigger_word: str | None = None
    # The original zip URL — kept for local testing only; never fetched at
    # runtime (the file is pre-staged in the read-only cache).
    dataset_zip: str | None = None

    @property
    def cached_model_dir(self) -> str:
        # Base model pre-staged here, keyed by a filesystem-safe form of --model.
        return f"/cache/models/{self.model.replace('/', '--')}"

    @property
    def cached_zip_path(self) -> str:
        # get_image_training_zip_save_path: /cache/datasets/{task_id}_tourn.zip
        return f"/cache/datasets/{self.task_id}_tourn.zip"

    @property
    def dataset_images_dir(self) -> str:
        # ai-toolkit dataset format: a FLAT folder of image + same-basename .txt.
        return "/dataset/images"

    @property
    def dataset_holdout_dir(self) -> str:
        # Kept outside the training folder.  The scorer expands these reserved
        # pairs into its own nonce-scoped probe dataset after training. Hashing
        # the untrusted task id also prevents path traversal and cross-task
        # contamination in a reused container.
        task_key = hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:12]
        return f"/dataset/forge-holdout-{task_key}"

    @property
    def training_folder(self) -> str:
        # process[0].training_folder — ai-toolkit writes {this}/{config.name}/.
        return f"/app/checkpoints/{self.task_id}"

    @property
    def save_root(self) -> str:
        # What the validator uploads: training_folder/{config.name} where
        # config.name == expected_repo_name. LoRA files land here FLAT.
        return f"/app/checkpoints/{self.task_id}/{self.expected_repo_name}"

    @property
    def output_dir(self) -> str:
        # cli/telemetry call spec.output_dir; alias it to save_root (flat, no
        # /checkpoint subfolder — that was the SDXL layout).
        return self.save_root

    @property
    def config_path(self) -> str:
        return f"/dataset/configs/{self.task_id}.yaml"

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        model: str,
        model_type: str,
        expected_repo_name: str,
        trigger_word: str | None,
        dataset_zip: str | None,
    ) -> "ImageSpec":
        return cls(
            task_id=task_id,
            model=model,
            model_type=(model_type or "").strip().lower(),
            expected_repo_name=expected_repo_name,
            trigger_word=(trigger_word or None),
            dataset_zip=dataset_zip,
        )
