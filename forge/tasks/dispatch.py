"""Map current image model types to the shared ai-toolkit training handler.

Anything outside the five validator-supported types returns ``None`` so the CLI
degrades to the fallback floor instead of forfeiting.
"""

from __future__ import annotations

from collections.abc import Callable

from forge.clock import Deadline
from forge.data.schema import ImageSpec

Handler = Callable[[ImageSpec, Deadline], None]


def for_model_type(model_type: str) -> Handler | None:
    if model_type in ("flux", "krea2", "ideogram4", "z-image", "qwen-image"):
        from forge.tasks.aitoolkit import run

        return run
    return None  # unknown future/retired type → fallback floor
