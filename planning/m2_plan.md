# M2 Planning Document
## Energy-Aware Expert Routing in Mixture-of-Experts Models

**Milestone:** M2 — Baselines, Prototype, and First Pareto Signal
**Owner:** Hrishabh Gupta
**Duration:** 2 weeks (immediately following M1 submission)
**Status:** Draft — finalize after M1 sign-off

---

## 0. Context and Entry Criteria

M1 delivered a 15-paper literature review, a comparison table across five routing strategies, a clearly articulated research gap (`logits = x·W_g − λ·e_gpu`), and three identified baselines (Switch, Random, Dense). M2 converts that paper plan into running code and the first empirical signal.

M2 begins only after M1 is submitted. Do not start baseline engineering before the review memo, visual guide, and comparison spreadsheet are frozen.

**Entry checklist:**
- M1 deliverables submitted and acknowledged
- GPU development environment reproducible from `environment-gpu.yml`
- `pynvml` import verified against the target GPU driver version
- PyTorch CUDA build and the T5 tokenizer (`google-t5/t5-small`) installed and smoke-tested

---

## 1. M2 Goals (Stretch Scope)

The goal of M2 is to produce three things, in order of decreasing priority:

1. **Three reproduced baselines on a shared tiny-MoE testbed.** Same data, same compute budget, same evaluation harness. This is the control plane for every future experiment.
2. **A minimal energy-aware routing prototype.** pynvml-in-the-loop gating on the same tiny-MoE, configurable via a single hyperparameter λ. This de-risks the core idea before M3 scales it up.
3. **A first Pareto sweep.** Small grid over λ producing a quality-vs-joules scatter plot. This is the earliest empirical signal the research question is answerable at all, even if the numbers are ugly.

Stretch scope means the third item is a hard deliverable, not a nice-to-have. Slipping it pushes M3 back.

---

## 2. Shared Testbed Specification

Before any baseline is trained, freeze the testbed. Every baseline and the prototype must run on this exact configuration, otherwise comparisons are meaningless.

**Model.** A tiny decoder-only transformer: 6 layers, d_model = 256, 4 attention heads, FFN hidden = 1024. MoE layers replace the FFN in layers 2 and 4 (sandwich pattern). E = 4 experts per MoE layer. This sizing is chosen so one full training run finishes inside a single M2 day on a single consumer-grade GPU. **Implementation must be minimal pure PyTorch** — no DeepSpeed-MoE or other distributed frameworks. The gating function and MoE layer must be hand-written so every line is understood. (Per professor feedback: "A minimal PyTorch implementation is better than a massive framework for testing this concept.")

**Data.** C4 `en.noclean` split via HuggingFace `datasets`, with a 1% random subsample saved to disk for rapid iteration. WikiText-2 for held-out perplexity. Fixed random seed, fixed token budget (target: ~50M tokens per run; adjust down if time-boxed).

**Evaluation harness.** A single `evaluate.py` that ingests a trained checkpoint and emits: (a) validation perplexity, (b) total wall-clock joules from pynvml integration over the evaluation loop, (c) mean router entropy, (d) load-balance loss at convergence. All four numbers go into a single JSON row per run. No bespoke logging per baseline.

**Energy measurement.** pynvml sampled at 10 Hz during training and evaluation. Trapezoidal integration over power readings gives joules. Baseline measurement overhead must be <3% of wall-clock; profile this before starting the sweep.

---

## 3. Baseline Reproduction Plan

### Baseline 1: Switch Transformer (λ = 0)

k=1 routing, single load-balance loss with coefficient α = 0.01 (per Fedus 2022). This is the primary baseline and the most important number in M2. Every other result is measured as a delta against this.

**Definition of done:** validation PPL within ±10% of a reference Switch implementation at matched token budget, router entropy non-degenerate (not collapsing to a single expert), load-balance loss stable in the last 10% of training.

### Baseline 2: Random Router

Tokens uniformly assigned to experts at every forward pass. No learned gating, no auxiliary loss. This is a sanity baseline: if Switch does not meaningfully beat Random on PPL, the MoE container has a bug and everything downstream is invalid.

**Definition of done:** PPL strictly worse than Switch by a visible margin. If it is not, stop and debug the container before proceeding.

### Baseline 3: Dense Transformer

Same d_model, same depth, FFN unchanged, no MoE layers. Parameter count will be lower than the MoE variant; that is expected and documented. The purpose is to establish the Patterson 2021 energy comparison reference point.

**Definition of done:** PPL reasonably close to Switch (dense with similar FLOPs should be competitive at this scale), joules-per-token strictly higher than Switch (confirming the sparse energy advantage the project builds on).

### Engineering notes

**Pure PyTorch, no DeepSpeed.** The professor explicitly advised against using a massive framework for testing this concept. Write the `TopKGating` module, `MoELayer`, and `Expert` (a simple 2-layer FFN) by hand. The custom gating function `g_i = softmax(x^T w_i − λ·e_i)` and the multi-objective loss must be written and fully understood by the researcher — these are the core of the research contribution and cannot be treated as a black box. Claude can help with boilerplate (data loading, training loop scaffolding, logging), but the gating + loss code is yours.

Write one `train.py` with a `--router` flag that dispatches to {switch, random, dense, energy_aware}. Do not fork the training loop per baseline — divergence between copies is the single biggest reproducibility risk.

Commit the exact seed, the exact config hash, and the exact git commit to the results JSON. Runs that cannot be reproduced do not count.

---

## 4. Energy-Aware Prototype

### 4.1 Gating function

Replace the standard top-k logits with:

```
logits = x · W_g − λ · e_gpu
top_idx = topk(logits, k=1)
```

where `e_gpu` is a length-E vector of current per-expert energy costs. For the prototype, all experts on a single GPU share the same `e_gpu` value (the instantaneous device power reading in watts, normalized by a fixed scale). This is deliberately the simplest possible instantiation — heterogeneous per-device energy enters in M3.

### 4.2 Loss

```
L = L_task + α · L_balance + β · L_energy
```

`L_energy` is the mean of the routed experts' energy costs over the batch. β is the training-time pressure on energy-efficient routing; λ is the inference-time penalty in the logits. Both matter, and they are not the same knob. The prototype uses β = 0 initially (only λ active) to isolate the gating-time effect before adding training-time pressure.

### 4.3 pynvml integration strategy

Polling pynvml synchronously on every forward pass will serialize the training step and tank throughput. Two-stage strategy:

**Background sampler thread** reads `nvmlDeviceGetPowerUsage` at 10 Hz into a ring buffer. The forward pass reads the most recent value without blocking. Accept up to ~100ms staleness; the energy state does not change that fast at the time scale of a training step.

**Profile the overhead first.** Before plugging the sampler into the router, run the Switch baseline with the sampler thread attached but not consumed. Measure steps/sec with and without. If overhead > 5%, redesign before continuing.

### 4.4 Definition of done for the prototype

The prototype "works" in the M2 sense if all of the following hold:

- Training is numerically stable for the full token budget
- λ > 0 produces measurably different routing distributions than λ = 0 on the same seed
- Total joules for the run strictly decrease as λ increases across the sweep range
- PPL degradation is monotonic in λ (no pathological non-monotonic collapses)

It does *not* need to beat Switch on a joint metric yet. M2 establishes that the mechanism is operational and well-behaved; M3 tunes it for wins.

---

## 5. Pareto Sweep

Small grid over λ ∈ {0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0}. Seven runs on the tiny testbed. Each run produces one (PPL, joules) point.

**Plot:** joules on the x-axis, PPL on the y-axis, log scale on joules, connect the λ-ordered points with a line. Overlay Switch (λ = 0), Random, and Dense as fixed reference dots.

**Reading the plot:** what you want to see is a convex frontier where increasing λ trades PPL for joules gracefully. What you want to *not* see is either (a) a flat line (λ has no effect — bug), or (b) a cliff (λ crushes quality before saving meaningful energy — wrong formulation).

**Exit criterion for M2 is the existence of this plot with interpretable structure**, not a particular shape. Even a bad frontier is a valid M2 outcome because it tells M3 exactly what to fix.

---

## 6. StructuredDNA Triage (dedicated section)

arXiv:2512.08968 — "A Bio-Physical Framework for Energy-Aware Transformer Routing" — surfaced during the M1 alert window as potentially competing prior art. M2 includes a bounded, time-boxed full read with a predetermined decision tree.

**Time box:** 0.5 day maximum for the full read and the triage writeup. If it blows past that, fall back to the abstract-only assessment.

### 6.1 The three criteria

For each criterion, answer strictly yes/no with the supporting quote and page number from the paper.

**Criterion 1 — Differentiable and learned end-to-end.**
Does the router have learnable parameters that are updated by the task loss during training? A pre-defined closed-form scoring function (even a clever one) is NO. A learned gating network with an energy regularizer is YES.

**Criterion 2 — Live hardware telemetry.**
Does the energy signal come from a real-time measurement of the physical device during inference/training, or is it an a priori cost model (static FLOPs estimate, per-expert constant, analytical energy formula)? A priori is NO. pynvml / RAPL / equivalent telemetry is YES.

**Criterion 3 — Integrated into the training loss.**
Is the energy term present in the backward pass, so that weights co-adapt to energy-aware routing during training? Inference-only heuristics applied on a pre-trained model are NO. A `+ β · L_energy` term in the training objective is YES.

### 6.2 Decision tree

**Case A: all three are NO.** StructuredDNA is related work on a different axis (heuristic energy-aware inference). Cite it in the related-work section, note the distinction clearly, no changes to the gap statement or methodology. This is the expected case given the signal you already have.

**Case B: exactly one is YES.** Partial collision. Refine the gap statement to emphasize the remaining two axes. Add StructuredDNA as a fourth baseline in M3 if feasible. Gap is still defensible.

**Case C: two are YES.** Significant collision. Pause M3 planning, write a half-page memo explaining how our contribution still differs (e.g., formulation, empirical regime, scale, multi-GPU heterogeneity). Bring to advisor review before proceeding.

**Case D: all three are YES.** Hard collision on novelty. Treat StructuredDNA as the new state of the art. Pivot the contribution narrative from "first energy-aware learned router" to one of: (a) a stronger empirical study on larger scale, (b) a multi-GPU heterogeneity extension, (c) an ablation-driven analysis of what makes energy-aware routing work. Escalate immediately.

### 6.3 Triage deliverable

A one-page markdown file in `planning/structureddna_triage.md` containing: the three yes/no answers with quotes, the selected case (A/B/C/D), and the resulting action. This file is the audit trail for the novelty claim.

---

## 7. Risks for M2

**pynvml sampling overhead.** Risk: the sampler thread serializes training and bloats wall-clock. Mitigation: profile before integration (section 4.3), cap at 5% overhead, fall back to lower sampling rate if needed. Do not start the sweep until this is measured.

**Baseline divergence.** Risk: the three baselines produce un-comparable numbers because the training loops drifted. Mitigation: one `train.py`, one `evaluate.py`, one config schema, one seed policy. Code review the diff between baselines before running.

**StructuredDNA collision.** Risk: the triage reveals Case C or D, invalidating part of the novelty claim. Mitigation: explicit decision tree in section 6.2, early escalation path, the three-axis contribution framing means at most one axis can collapse.

**Token budget underpowered.** Risk: 50M tokens is too few to separate signals; all frontier points look the same within noise. Mitigation: report error bars from three seeds on a subset of λ values; if noise dominates, document it and raise token budget in M3.

**pynvml driver mismatch.** Risk: pynvml version and GPU driver version disagree, silently returning stale or zero power readings. Mitigation: smoke test on day 1 — read power with a known GPU load and confirm the number moves.

---

## 8. Week-by-Week Schedule

**Week 1**

*Day 1.* Environment smoke test, pynvml sanity check, testbed config frozen, `train.py` skeleton with `--router` flag.
*Day 2.* Switch baseline trained, evaluation harness producing the four numbers into JSON.
*Day 3.* Random and Dense baselines trained. Sanity check: Switch < Random on PPL; Dense joules-per-token > Switch.
*Day 4.* pynvml sampler thread implemented, overhead profiled, prototype gating function wired up.
*Day 5.* Prototype trained at λ = 0 (sanity: should reproduce Switch). Prototype trained at one non-zero λ (sanity: routing changes). Buffer / catch-up.

**Week 2**

*Day 6.* StructuredDNA full read, triage doc written, decision made.
*Day 7.* Full λ sweep launched (7 runs; parallelize if possible, sequential otherwise).
*Day 8.* Sweep completes, Pareto plot generated, interpretation written.
*Day 9.* M2 writeup: results section, updated gap statement if needed, figures.
*Day 10.* Buffer, advisor review, M2 submission.

Day 10 exists because something will go wrong on day 4 or day 7 and the whole week will shift. Do not schedule real work into day 10.

---

## 9. M2 Exit Criteria

M2 is complete when all of the following are true:

- Three baselines reproduced on the shared testbed with results in `results/m2/baselines.json`
- Energy-aware prototype training runs cleanly at three or more λ values
- Pareto plot exists in `figures/m2/pareto_lambda_sweep.png` with interpretable structure
- StructuredDNA triage document exists with a selected case and an action
- A short M2 memo (one page) summarizes what was learned and what M3 should change

If any of these is missing on day 10, do not declare M2 done — reduce M3 scope instead of faking M2 closure.

---

## 10. Open Questions for Advisor Review

Before starting M2, get explicit sign-off on:

1. Is a single-GPU testbed acceptable for M2, with multi-GPU heterogeneity deferred to M3?
2. Is 50M tokens sufficient for the sweep, or should a larger budget be allocated at the cost of scope?
3. If StructuredDNA triage hits Case C or D, what is the expected pivot direction?
4. Should the M2 writeup be submitted as a standalone memo or folded into the M3 intro?
