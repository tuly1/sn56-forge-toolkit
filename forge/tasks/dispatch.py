"""Map a model type to its training handler.

The 3 implemented ai-toolkit types route to the ai-toolkit trainer; anything else
(z-image / qwen-image / unknown) returns None so the CLI degrades to the fallback
floor instead of forfeiting.
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
    return None  # z-image / qwen-image / unknown → fallback floor
