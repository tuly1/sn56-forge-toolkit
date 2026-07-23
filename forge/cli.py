"""Entry point for the ai-toolkit image trainer. Parses the validator-supplied
image-task arguments and dispatches to the handler for the model type. Kept
deliberately thin: all real work lives in the task modules.

Guiding rule: never exit non-zero. The validator treats a non-zero exit before
the wall-clock kill as a failure with no upload (scored -1), whereas any model
left at the output path is uploaded and scored. So every failure path funnels
into the fallback, which guarantees a valid artifact when any save exists.
"""

from __future__ import annotations

import argparse
import sys
import time

from forge import telemetry
from forge.clock import Deadline
from forge.data.schema import ImageSpec

# Reserve a slice of the wall clock for final export so a kill never catches us
# mid-write. ai-toolkit's flush on SIGTERM is comparable to kohya's, so we keep
# the same 180s reserve.
_EXPORT_RESERVE_SECONDS = 180


def _parse(argv: list[str]) -> argparse.Namespace:
    # Nothing is `required=` and hours stays a string: argparse's own failure
    # mode is sys.exit(2), which is the one thing the never-forfeit contract
    # forbids. Missing/malformed values are handled in main instead.
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--task-id", default=None)
    p.add_argument("--model", default=None)
    # --dataset-zip is the original URL; the file is pre-staged in the read-only
    # cache, so we keep the arg only for local testing and never fetch it.
    p.add_argument("--dataset-zip", default=None)
    # No `choices=`: accept any value and let dispatch decide, so an unseen model
    # type falls through to the fallback instead of an argparse exit(2).
    p.add_argument("--model-type", default=None)
    p.add_argument("--expected-repo-name", default=None)
    p.add_argument("--hours-to-complete", default=None)
    p.add_argument("--trigger-word", default=None)
    try:
        known, _unknown = p.parse_known_args(argv)
        return known
    except BaseException:  # noqa: BLE001 — includes SystemExit on malformed tokens
        return argparse.Namespace(
            task_id=None, model=None, dataset_zip=None, model_type=None,
            expected_repo_name=None, hours_to_complete=None, trigger_word=None,
        )


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = _parse(sys.argv[1:] if argv is None else argv)

    if not args.task_id or not args.expected_repo_name:
        # Without these two we cannot even address save_root — exit 0 so the
        # container never reports the hard failure an argparse exit(2) would.
        _log("missing --task-id/--expected-repo-name; nothing to do")
        return 0
    try:
        hours = float(args.hours_to_complete)
    except (TypeError, ValueError):
        hours = 1.0  # keep the deadline machinery alive on a malformed value

    deadline = Deadline.from_hours(
        hours,
        started_monotonic=started,
        export_reserve_s=_EXPORT_RESERVE_SECONDS,
    )
    telemetry.start_run(
        task_id=args.task_id,
        model_type=args.model_type,
        model_arg=args.model,
        hours_to_complete=hours,
        trigger_word=args.trigger_word,
    )

    try:
        spec = ImageSpec.build(
            task_id=args.task_id,
            model=args.model,
            model_type=args.model_type,
            expected_repo_name=args.expected_repo_name,
            trigger_word=args.trigger_word,
            dataset_zip=args.dataset_zip,
        )
    except BaseException as exc:  # noqa: BLE001
        _log(f"spec build failed ({type(exc).__name__}: {exc}); bare spec + fallback")
        telemetry.event("spec_build_failed", error=f"{type(exc).__name__}: {exc}")
        spec = ImageSpec(
            task_id=args.task_id,
            model=args.model,
            model_type=(args.model_type or "").lower(),
            expected_repo_name=args.expected_repo_name,
        )

    _run(spec, deadline)
    return 0


def _run(spec: ImageSpec, deadline: Deadline) -> None:
    """Dispatch on model type, degrading to the fallback on any failure. We import
    the handler lazily so heavy diffusion deps don't load for a model type this
    build doesn't implement, and we catch everything.
    """
    # Replace any recorder left by an older build before fallible setup.  The
    # compatibility writer stores full telemetry only in container-local /tmp
    # and puts a strict public projection in the exact validator upload root.
    try:
        import os

        os.makedirs(spec.output_dir, exist_ok=True)
        telemetry.prepare_public_recorder(spec.output_dir)
    except Exception:
        pass

    # Scope checkpoint discovery before dispatch so even import failures and
    # unknown future model types cannot promote stale files from a prior retry.
    try:
        from forge.tasks import checkpoints

        scope = checkpoints.begin_run(
            spec.save_root,
            spec.expected_repo_name,
            task_id=spec.task_id,
        )
        telemetry.bind_private_bundle(
            spec.output_dir, str(scope.get("attempt_nonce") or "")
        )
        telemetry.write_into(spec.output_dir)
    except Exception as exc:
        telemetry.event(
            "checkpoint_scope_failed", error=f"{type(exc).__name__}: {exc}"
        )

    handler = None
    try:
        from forge.tasks import dispatch

        handler = dispatch.for_model_type(spec.model_type)
    except Exception as exc:  # dispatch import problems must not forfeit
        _log(f"dispatch failed for {spec.model_type!r}: {exc!r}")
        telemetry.event("dispatch_failed", error=repr(exc))

    if handler is not None:
        try:
            handler(spec, deadline)
            telemetry.event("run_complete")
            _finalize_public_bundle(spec)
            return
        except BaseException as exc:  # noqa: BLE001
            _log(f"handler raised ({type(exc).__name__}: {exc}); using fallback")
            telemetry.event("handler_failed", error=f"{type(exc).__name__}: {exc}")

    try:
        from forge.tasks.fallback import emit_untrained_copy

        emit_untrained_copy(spec)
    except Exception as exc:
        _log(f"fallback failed: {exc!r}")
        telemetry.event("fallback_failed", error=repr(exc))
    _finalize_public_bundle(spec)


def _finalize_public_bundle(spec: ImageSpec) -> None:
    """Terminal upload construction; diagnostics must never affect exit status."""
    try:
        from forge.tasks import publication

        publication.finalize_public_bundle(spec.output_dir)
    except BaseException as exc:  # noqa: BLE001
        telemetry.event(
            "public_bundle_failed", error=f"{type(exc).__name__}: {exc}"
        )
        telemetry.write_into(spec.output_dir)


def _log(msg: str) -> None:
    print(f"[forge.cli] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
