# Aug-10 `3bd1ebed` early-exit and cadence record

Status: observational field forensics plus a recovery/observability patch. This
record does **not** identify the child-process root cause and does **not** make a
quality claim for denser checkpoints.

## What the retained evidence proves

- The task API records a one-hour Krea2 task.
- The public Forge recorder records `toolkit_start` at 5.7 seconds,
  `toolkit_end` at 2425.2 seconds, and successful checkpoint finalization at
  2425.7 seconds.
- Exact config replay plans 1,672 steps with the incumbent 335-step save
  cadence.
- The public repository contains periodic artifacts at 335, 670, 1005, and
  1340. `last.safetensors` is byte-identical to step 1340; the unnumbered exact
  final is absent.
- The served runner's one-hour deadline path would start termination at about
  3375 seconds: 3,600 seconds minus the 180-second export reserve and the
  45-second boundary margin. The child ended about 950 seconds before that
  frontier. The parent remained alive, finalized the step-1340 artifact, and
  uploaded it.

This rules out planned completion, Forge's own deadline stop, and a whole-
container death. It supports an unexpected child exit or child-only
termination. The exact trigger remains unknowable: the public projection did
not retain the return-code/deadline class, and the hash-bound private record
(`9757f8d77ed97c9e76a20d2b526e7e569019860ac7229ba5f987f7a0e25ee7ee`)
and toolkit log did not survive the ephemeral container.

## Cadence attribution

The planned-to-shipped difference is exactly 332 steps, but it is wrong to
attribute all 332 to save cadence. A two-run throughput fit estimates that the
child may have reached roughly step 1,489 before exiting; under that estimate,
the 335-step cadence discarded about 149 completed-but-unsaved steps. That
step-1,489 value is an estimate, not an observed counter.

The code change therefore makes only the bounded claim: Krea's maximum unsaved
interval falls from 334 steps on this plan to 199 steps. At a 1,672-step plan,
the new 200-step cadence exposes eight periodic candidates (200 through 1600)
instead of four. It does not change steps, loss, learning rate, runtime, or any
non-Krea config.

The closest retained H100 measurement for the same 228.6 MB Krea LoRA family
estimates about 2.5 seconds incremental latency per periodic save across four
saves. Four additional saves therefore project to roughly 10 seconds on that
host. This is an estimate from a prior run, not a target-host timing
certificate; a production H100 smoke remains required before release.

## Observability contract

The public recorder now emits one bounded event name after the toolkit exits:

- `toolkit_exit_zero_{present|absent}`
- `toolkit_exit_deadline_{present|absent}`
- `toolkit_exit_nonzero_{present|absent}`
- `toolkit_exit_signal_{present|absent}`
- `toolkit_exit_unknown_{present|absent}`

`present` means only that a current-run filename and filesystem identity were
observed. It deliberately does not claim that the bytes are structurally valid
or that promotion succeeded. The later `checkpoint_finalized` event proves
only that finalization yielded a usable artifact; the public projection does
not reveal whether that artifact came from the observed current-run candidate
or a preserved prior fallback. A truncated-file regression pins this
distinction. Raw return codes, signal numbers, and log text remain private.
Never-forfeit behavior is unchanged: an unexpected nonzero exit with a valid
current-run LoRA is still finalized; an unexpected nonzero exit with no
current-run checkpoint entry raises immediately into the existing fallback
path; a present but invalid entry fails the later finalizer and reaches that
same fallback without a false public salvage claim.

## Evidence anchors

- Public recorder CAS object SHA-256:
  `2225ba415ac2805a38f43a37987a96ebd2ce983654bf16e960ad564ca6273adc`
- Aug-10 loss-forensics document SHA-256:
  `196f61ea91bad316c55d24a64907898465fc8bcf4bec6b04699e276e3ef95bc2`
- H100 terminal loss database SHA-256:
  `665b82e0604c1ad1e373a86c64c052df2b9152ce2abd1e7f62f2d85642286e1c`
- Base commit: `4152f4cd4463df5bc083f534423240e9514c2750`

The corresponding local evidence paths and the inspection-side-effect note are
carried in the project handoff, outside the public trainer repository.
