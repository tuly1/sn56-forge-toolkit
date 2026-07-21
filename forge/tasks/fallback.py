"""Last-resort, run-scoped output for ai-toolkit image tasks.

The fallback promotes a valid checkpoint produced by the current attempt.  If
there is none, it may retain an intact ``last.safetensors`` that predated the
attempt, but records that outcome explicitly.  Unchanged periodic files are
never treated as current output, preventing same-task retry contamination.

We still cannot synthesize a valid untrained ai-toolkit LoRA without knowing the
architecture-specific network-key skeleton.  With no current save and no valid
prior ``last.safetensors``, there is nothing scoreable to emit.
"""

from __future__ import annotations

from forge.data.schema import ImageSpec
from forge.tasks import checkpoints


def emit_untrained_copy(spec: ImageSpec) -> None:
    from forge import telemetry

    record = checkpoints.finalize(
        spec.save_root,
        spec.expected_repo_name,
        context="cli_fallback",
    )
    if record is None:
        telemetry.event("fallback_no_current_or_prior_checkpoint")
    elif record["status"] == "selected_current_run":
        telemetry.event(
            "fallback_promoted_current_checkpoint",
            source=record["source"],
            selected_step=record["selected_step"],
        )
    else:
        telemetry.event("fallback_retained_previous_run_checkpoint")
