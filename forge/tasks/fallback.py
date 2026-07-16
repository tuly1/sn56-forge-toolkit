"""Last-resort output for ai-toolkit image tasks.

A non-zero exit uploads nothing and scores -1; any valid LoRA at
``{save_root}/last.safetensors`` gets uploaded and scored. The primary
kill-safety is ai-toolkit's periodic ``save_every`` saves, so this fallback's main
job is to promote whatever checkpoint exists to the evaluator's preferred
filename.

Honest limitation (same as the SDXL floor): we cannot synthesise a *valid*
untrained ai-toolkit LoRA blind (the network-key skeleton must match ComfyUI's
loader), so if ai-toolkit failed before its first save there is no floor to emit —
that task scores -1.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

from forge.data.schema import ImageSpec
from forge.tasks.integrity import valid_safetensors


def emit_untrained_copy(spec: ImageSpec) -> None:
    from forge import telemetry

    root = spec.save_root
    loras = [
        p for p in _safetensors(root) if os.path.basename(p) != "last.safetensors"
    ]
    if loras:
        # A periodic ai-toolkit checkpoint exists — make sure last.safetensors is
        # present so the evaluator's preferred lookup hits it FIRST.
        last = os.path.join(root, "last.safetensors")
        if not os.path.isfile(last):
            # Newest checkpoint may be TRUNCATED (non-atomic ai-toolkit save cut
            # by the deadline kill) — step down to the newest one that passes an
            # integrity check rather than promoting a corrupt submission.
            ordered = sorted(loras, key=_step_of, reverse=True)
            src = next((p for p in ordered if valid_safetensors(p)), ordered[0])
            # Atomic tmp+replace: a mid-copy kill onto last.safetensors (the
            # evaluator's first-matched file) would leave a truncated submission
            # preferred over the intact periodic checkpoints → zero score.
            tmp = last + ".tmp"
            try:
                shutil.copy(src, tmp)
                os.replace(tmp, last)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        telemetry.event("fallback_kept_checkpoint", files=len(loras))
        return
    telemetry.event("fallback_no_checkpoint")  # nothing scoreable — accepts -1


def _safetensors(path: str) -> list[str]:
    return glob.glob(os.path.join(path, "*.safetensors")) if os.path.isdir(path) else []


def _step_of(path: str) -> int:
    m = re.search(r"_(\d+)\.safetensors$", os.path.basename(path))
    return int(m.group(1)) if m else -1
