"""Step / save-cadence policy — OUR EDGE.

The eval scores 0.25*caption-guided + 0.75*UNCONDITIONAL L2, and over-training
PUNISHES the unconditional term, so the whole edge is completing training near
the optimum rather than blindly running the flat 2000-step template. We scale
steps sublinearly with dataset size (a power law, like the champion's z-image /
qwen laws) and then cap by a conservative wall-time budget so we finish rather
than get killed. Everything here is one tunable table + never raises.
"""

from __future__ import annotations

# --- v1 step-scaling table (MEASURE + tune from the eval harness) ----------
# steps = clamp(base * (N / n_ref)**p, min, max)
# Grounded in champion ai-toolkit laws (z-image base~1100 p~0.5, qwen base~1000
# p~0.5, subject-tighter p~0.85). flux = well-understood LoRA → a more aggressive
# cut from the flat template 2000. krea2/ideogram4 = BRAND NEW, no incumbent →
# CONSERVATIVE: base near the template, only a modest reduction for tiny sets, and
# max capped at the template so we never over-train past the field's only anchor.
_N_REF = 24  # ~mid of the 10-50 pair range
# krea2/ideogram4: the 0.75-weighted UNCONDITIONAL L2 punishes depth INDEPENDENT
# of incumbency (champion effective steps ~200-1000; over-training is the #1
# liability), so the earlier "stay near the template 2000" stance over-scheduled
# the tiny/mid buckets. Base pulled 1500→1200 (still a hair above flux's 1100 to
# hedge the brand-new archs), the 800 floor dropped to 400 so tiny sets aren't
# forced deep, and 2000 kept only as a hard ceiling. Lock via eval-harness A/B.
STEP_TABLE = {
    #            base  n_ref   p    min    max
    "flux": dict(base=1100, n_ref=_N_REF, p=0.50, min=500, max=2000),
    # krea2/ideogram4 RE-CALIBRATED Jul 16 on real photos w/ true holdout,
    # scored by the validator's own stack (10-seed img2img, 0.25 text +
    # 0.75 unconditional MSE):
    # - ideogram4 @ lr 1e-4: score beats zero-LoRA base from ~32 steps on,
    #   best at ~96 steps for 12 imgs (0.0290 vs base 0.0351); deep training
    #   never helped. base 140 @ n_ref 24 ≈ 96 @ 12 under p=0.5.
    # - krea2: near-flat score across 8..128 steps at both tested LRs (arch
    #   trains very slowly from a near-converged objective) — fewer steps =
    #   same score, faster finish, less over-train risk.
    "krea2": dict(base=300, n_ref=_N_REF, p=0.50, min=100, max=400),
    "ideogram4": dict(base=140, n_ref=_N_REF, p=0.50, min=48, max=400),
    # z-image/qwen: straight from the champion's published power laws
    # (z-image base~1100 p~0.5; qwen base~1000 p~0.5). qwen's template is 3000
    # steps and its tasks get a +0.5h grant, so its ceiling stays the template.
    "z-image": dict(base=1100, n_ref=_N_REF, p=0.50, min=400, max=2000),
    "qwen-image": dict(base=1000, n_ref=_N_REF, p=0.50, min=400, max=3000),
}

# --- v1 wall-time budget model (CONSERVATIVE; measure on H100) --------------
# H100, grad-checkpointing, res<=1024, batch 1. Upper-bound s/it so we never
# over-schedule. krea2's do_differential_guidance adds a second guidance forward
# per step → highest. These are guesses to be replaced by measured per_step().
# z-image/qwen train quantized (qfloat8/uint3, low_vram) → slower per it;
# conservative until measured.
SEC_PER_IT = {"flux": 2.5, "krea2": 3.5, "ideogram4": 3.0,
              "z-image": 3.0, "qwen-image": 4.0}
STARTUP_S = 300.0  # big base-model load + latent/text-embed warmup
EXPORT_RESERVE_S = 180.0  # mirrors cli._EXPORT_RESERVE_SECONDS
MARGIN = 0.85  # jitter/save/eval headroom


def size_scaled_steps(model_type, num_images, hours_to_complete, template_steps):
    """Never raises → falls back to template_steps on any error (INV-1)."""
    try:
        mt = (model_type or "").strip().lower()
        row = STEP_TABLE.get(mt)
        if row is None:
            return int(template_steps)
        n = max(1, int(num_images))
        scaled = row["base"] * (n / row["n_ref"]) ** row["p"]
        scaled = int(round(max(row["min"], min(row["max"], scaled))))

        sit = SEC_PER_IT.get(mt, 3.0)
        budget_s = max(0.0, float(hours_to_complete) * 3600.0)
        train_s = budget_s * MARGIN - STARTUP_S - EXPORT_RESERVE_S
        budget_cap = int(train_s / sit) if train_s > 0 else 1
        return max(1, min(scaled, budget_cap))  # cap may push below `min`
    except Exception:
        try:
            return int(template_steps)
        except Exception:
            return 1000


def kill_safe_save_every(steps, template_save_every):
    """Cadence for periodic {repo}_{step}.safetensors saves — the sole kill-safety
    mechanism (INV-2). The FIRST save must land EARLY: the deadline monitor (or a
    crash) can cut a short/slower-than-modeled run before the template's ~200-250
    cadence, leaving _finalize with no LoRA → forfeit. We cap at 100 (and steps//8)
    so a scoreable checkpoint exists well before 25% of the schedule, at negligible
    I/O (max_step_saves_to_keep caps retained files). Floored at 1. Never raises.
    """
    try:
        return max(1, min(int(template_save_every), max(1, int(steps) // 8), 100))
    except Exception:
        try:
            return int(template_save_every)
        except Exception:
            return 100
