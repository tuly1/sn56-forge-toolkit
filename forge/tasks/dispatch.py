"""Map image model types to their validator-routed training handler.

Anything outside the five validator-supported types returns ``None`` so the CLI
degrades to the fallback floor instead of forfeiting.
"""

from __future__ import annotations

from collections.abc import Callable
import os

from forge.clock import Deadline
from forge.data.schema import ImageSpec

Handler = Callable[[ImageSpec, Deadline], None]


def for_model_type(model_type: str) -> Handler | None:
    if model_type == "flux" and os.environ.get("FORGE_FLUX_BACKEND") == "kohya":
        # G.O.D routes FLUX to the legacy-named, dual-runtime Dockerfile. Its
        # shape-aware handler selects Kohya only for the downloader's normalized
        # exact-one-file cache and ai-toolkit for full snapshot directories. The
        # toolkit-named Dockerfile deliberately does not set this switch.
        from forge.tasks.flux_kohya import run

        return run
    if model_type in ("flux", "krea2", "ideogram4", "z-image", "qwen-image"):
        from forge.tasks.aitoolkit import run

        return run
    return None  # unknown future/retired type → fallback floor
