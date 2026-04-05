# Paper Learnings — M1 Literature Review

> Energy-Aware Expert Routing in Mixture-of-Experts Models
> Risk mitigation applied: **Triage reading** — abstract/conclusion first, then methods, full read only if highly relevant.
> Constraint: ~½ page per paper, ≤3 ASCII diagrams, duplicity flagged.

---

## ASCII Diagram 1 — MoE Gating Evolution Across Papers

```
  Papers [1-3,14,15]              Our Contribution
  ──────────────────              ────────────────────────────────
  Standard MoE Gate               Energy-Aware Gate

  token x                         token x
    │                               │
    ▼                               ▼
  ┌──────────┐                   ┌──────────┐
  │ x · W_g  │  (affinity)       │ x · W_g  │  (affinity)
  └────┬─────┘                   └────┬─────┘
       │                              │
       ▼                              ▼
  ┌──────────┐                   ┌────────────────────┐
  │ Top-k    │                   │ h(x) - λ · e_gpu   │ ◄── energy
  │ Softmax  │                   │ Top-k Softmax      │     penalty
  └────┬─────┘                   └─────────┬──────────┘
       │                                   │
       ▼                                   ▼
  route to k                         route to k experts
  experts                            (prefer low-power GPUs)

  Loss:                           Loss:
    L_task + α·L_balance            L_task + α·L_balance + β·L_energy
```

---

## ASCII Diagram 2 — Paper Taxonomy & Research Pillars

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   LITERATURE LANDSCAPE                       │
  ├──────────────────┬──────────────────┬────────────────────────┤
  │  MoE ARCH        │  MoE ANALYSIS    │  GREEN AI / ENERGY     │
  │  (how to route)  │  (what happens)  │  (how to measure)      │
  ├──────────────────┼──────────────────┼────────────────────────┤
  │ [1] Shazeer '17  │ [4] Riquelme '21 │ [7]  Schwartz '20      │
  │ [2] Fedus   '22  │ [5] Clark    '22 │ [8]  Patterson '21     │
  │ [3] Lepikhin'20  │ [6] Puigcerv.'23 │ [9]  Lannelongue '21   │
  │ [14] Mustafa '22 │                  │ [10] Henderson '20     │
  │ [15] Kool   '24  │                  │ [11] De Vogeleer '22   │
  │                  │                  │ [12] nvidia-smi '23    │
  ├──────────────────┴──────────────────┤                        │
  │  CROSS-CUTTING                      │ [13] Hooker '21        │
  │  Hardware constraints shape which   │  (Hardware Lottery)    │
  │  routing algorithms even get tried  │                        │
  └─────────────────────────────────────┴────────────────────────┘

  GAP: No paper in any column incorporates real-time, per-GPU
       energy costs INTO the routing decision function itself.
```

---

## ASCII Diagram 3 — Pareto Frontier Concept (RQ1 Target)

```
  Perplexity (PPL)
  ▲
  │
  │  x  Random Router [baseline-lower-bound]
  │
  │        x  Switch Transformer (λ=0) [baseline]
  │
  │           ·  λ=1e-5
  │            ·  λ=1e-4         ← Pareto frontier
  │              ·  λ=1e-3          (our target curve)
  │                 ·  λ=1e-2
  │                      ·  λ=1e-1
  │
  └──────────────────────────────────────►  Energy (Joules)
         lower ◄────────────► higher

  Goal: for small λ, nearly no PPL loss with measurable
  energy reduction; aggressive λ trades PPL for large savings.
```

---

## Pillar A — MoE Architecture Papers

### [1] Shazeer et al. (2017) — Outrageously Large Neural Networks: Sparsely-Gated MoE Layer
**Venue:** ICLR · **Relevance:** ★★★★★

Introduces the foundational MoE gating equation: `G(x) = Softmax(TopK(H(x), k))` with noisy top-k selection over N experts. Two auxiliary losses — importance loss (even gate-value sums) and load loss (even token counts) — prevent expert collapse. Scales to 137B parameters / 4096 experts on LSTM-based LM and MT. Results: 24% lower perplexity vs. compute-matched dense baseline on 1B-Word benchmark; +1.34 BLEU on WMT'14 En→Fr.

**Key insight for us:** The noisy additive term before softmax+top-k is the *exact architectural hook* we exploit — we replace Gaussian noise with `−λ · e_gpu`, making the energy penalty structurally analogous to the exploration noise. The importance + load losses are *preconditions* for our energy penalty: without balanced experts, energy-aware redistribution is meaningless.

**Limitations:** LSTM-based (not Transformer); no energy/power analysis; fragile load balancing.

---

### [2] Fedus et al. (2022) — Switch Transformers
**Venue:** JMLR · **Relevance:** ★★★★★

Simplifies MoE to k=1 (single expert per token) inside a Transformer, calling it the Switch layer. Trained on C4 — the same dataset we use. Achieves 7× pre-training speedup over T5-Base with the same compute; Switch-C is 4× faster to a fixed perplexity than T5-XXL. Introduces a simplified load-balancing loss `α · N · Σ(f_i · P_i)` and a capacity factor to handle expert buffer overflow.

**Key insight for us:** This is our **λ=0 baseline** — a standard Switch Transformer with no energy penalty. Direct perplexity comparison on C4 is possible. The k=1 routing simplifies energy accounting: each token's energy cost maps to exactly one expert's GPU.

**Duplicity note:** Shares gating fundamentals with [1] Shazeer; the load loss is a simplified variant of [1]'s importance+load pair. Capacity-factor overflow handling not present in [1].

---

### [3] Lepikhin et al. (2020) — GShard
**Venue:** ICLR · **Relevance:** ★★★★

Scales MoE to 600B parameters across 2048 TPUs for multilingual MT (100 languages → English). Uses top-2 expert routing (halfway between [1]'s top-k and [2]'s top-1). Introduces automatic compiler-level sharding (SPMD) that removes manual parallelism engineering. Trained in 4 days; achieves superior BLEU over prior art across 100 language pairs.

**Key insight for us:** Top-2 routing is our A4 ablation (`k=1` vs `k=2`). With k=2, the energy penalty can push the *second* expert choice toward a lower-power GPU while the primary expert handles quality — a softer energy-quality trade-off than k=1.

**Duplicity note:** Gating equation identical to [1]; top-2 is a specific instantiation. Automatic sharding is orthogonal to our routing modification but relevant to deployment.

---

### [14] Mustafa et al. (2022) — LIMoE (Multimodal-MoE)
**Venue:** NeurIPS · **Relevance:** ★★★

First large-scale multimodal MoE. Alternates dense and MoE layers for contrastive image-text learning. Proposes per-modality entropy-based regularization (local + global entropy losses) to stabilize training and prevent expert collapse. 84.1% zero-shot ImageNet (5.6B params, ~675M active per token). Emergent specialization: some experts become image-only, some text-only, some multimodal — without explicit constraints.

**Key insight for us:** Demonstrates that MoE routing is already multi-objective (accuracy + load balance); adding energy as a *third* objective is a natural extension. The entropy regularization technique is a candidate alternative to our additive penalty approach.

**Duplicity note:** Uses same top-k routing as [1-3]; entropy regularization is a novel addition not present in earlier MoE papers.

---

### [15] Kool et al. (2024) — MoE-Mamba
**Venue:** ICLR · **Relevance:** ★★★

Alternates Mamba (SSM) layers with MoE layers, combining linear-time inference (O(n)) with conditional computation. Uses Switch-style k=1 routing with load-balancing loss. Achieves the same performance as Mamba in 2.2× fewer training steps while preserving 5× inference throughput over Transformers.

**Key insight for us:** Shows MoE routing is architecture-agnostic (works beyond Transformers). The Mamba layers provide global sequence context *before* expert selection, which could enable more informed energy-aware routing. Training step reduction is a *proxy* for energy savings but not equivalent — validates the need for direct Joule measurement (our approach).

**Duplicity note:** Routing mechanism is essentially [2] Switch applied to SSM; novel contribution is the interleaving architecture, not the routing itself.

---

## Pillar B — MoE Analysis Papers

### [4] Riquelme et al. (2021) — V-MoE (Scaling Vision with Sparse MoE)
**Venue:** NeurIPS · **Relevance:** ★★★★

Applies sparse MoE to Vision Transformers. 15B-parameter V-MoE reaches 90.35% ImageNet (fine-tuned). Introduces Batch Priority Routing (BPR): tokens are scored, and the router processes high-priority tokens first, enabling adaptive per-image compute. Finding: experts tend to learn similar functions (low diversity), extracting common features across all classes rather than specializing.

**Key insight for us:** BPR's priority mechanism is conceptually adjacent to our energy penalty — both re-rank tokens before routing. The low expert diversity finding suggests many experts are *interchangeable*, which our energy penalty can exploit: swapping to a lower-power expert incurs minimal quality loss when experts are near-identical.

**Duplicity note:** Gating follows [1-2]; BPR is novel. Low-diversity finding aligns with [6] Puigcerver's single-expert result.

---

### [5] Clark et al. (2022) — What Do Experts Do in Sparsely-Gated MoE?
**Venue:** EMNLP · **Relevance:** ★★★★

Analyzes expert specialization in trained MoE language models. Finds that experts develop *soft* specialization: they have token-type preferences (punctuation, entities, function words) but are not strictly partitioned by linguistic category. Routing entropy decreases in deeper layers (more confident routing). Some experts become "generalists" handling diverse token types; others are narrow specialists.

**Key insight for us:** Directly informs RQ2. If "simpler" tokens (high routing confidence, low entropy) go to certain experts, our energy penalty (increasing λ) should preferentially route those tokens to low-power GPUs. The specialist vs. generalist distinction suggests differential energy sensitivity — generalist experts may tolerate the energy penalty better since they're already flexible.

**Duplicity note:** Builds on [1-2]'s architectures; provides the *analysis* that [1-2] lack. Complements [4]'s finding about expert similarity from a linguistic (not vision) angle.

---

### [6] Puigcerver et al. (2023) — From Sparse to Soft Mixtures of Experts
**Venue:** NeurIPS · **Relevance:** ★★★

Proposes Soft MoE: instead of hard top-k routing, each expert receives a *weighted combination* of all tokens (fully differentiable). Eliminates token dropping, expert collapse, and routing instability entirely. Soft MoE Huge/14 (128 experts) achieves competitive results with only 2% more inference time than dense ViT-H/14. Scales smoothly with expert count — no dead experts.

**Key insight for us:** Soft MoE is an *alternative paradigm* where our discrete energy penalty doesn't directly apply (no hard assignment to penalize). However, it provides an important **upper-bound comparison**: if soft routing achieves similar quality without energy awareness, our method's value is in the *hard-routing regime* where discrete expert selection can be energy-steered. Also validates that routing instability (which our λ perturbation could trigger) is a real concern in sparse MoE.

**Duplicity note:** Directly challenges the hard-routing paradigm of [1-3]. Expert collapse problem discussed in [1-2] is *solved* here but via a fundamentally different mechanism.

---

## Pillar C — Green AI & Energy Measurement Papers

### [7] Schwartz et al. (2020) — Green AI
**Venue:** Communications of the ACM · **Relevance:** ★★★★

Position paper distinguishing "Red AI" (accuracy at any compute cost) from "Green AI" (accuracy per unit of compute). Proposes FLOPs as a primary efficiency metric. Documents that NLP model training can emit up to 284,000 kg CO₂. Calls for reporting efficiency alongside accuracy in all ML publications.

**Key insight for us:** Establishes the *philosophical foundation* for our work. Our project is a concrete instantiation of Green AI: the λ parameter literally controls the Red↔Green trade-off. The Pareto frontier we plot (perplexity vs. Joules) is the empirical realization of Schwartz's efficiency reporting call.

**Limitations:** Proposes FLOPs as the metric, but FLOPs ≠ energy (hardware-dependent). We use Joules directly, which is more principled.

---

### [8] Patterson et al. (2021) — Carbon Emissions and Large Neural Network Training
**Venue:** arXiv · **Relevance:** ★★★★

Empirically quantifies energy/carbon for training large models (T5, GPT-3, Switch Transformer). Key finding: sparse models (Switch Transformer) consume <1/10th the energy of dense equivalents for the same quality. Reports concrete numbers: training T5-XXL costs ~86 MWh; Switch Transformer achieves similar quality at a fraction. Argues that model architecture, datacenter PUE, and grid carbon intensity all matter.

**Key insight for us:** Provides the *empirical evidence* that sparse MoE already saves energy vs. dense models. Our contribution goes further: not just "sparse is cheaper" but "energy-aware sparse routing is cheaper than energy-agnostic sparse routing." Patterson's numbers are our motivating data.

**Duplicity note:** Energy measurements complement [7]'s philosophical argument. Switch Transformer analysis overlaps with [2] Fedus but from an energy (not perplexity) perspective.

---

### [9] Lannelongue et al. (2021) — Green Algorithms
**Venue:** Advanced Science · **Relevance:** ★★★

Develops a standardized methodology and open-source calculator (green-algorithms.org) for estimating computation's carbon footprint. Factors in runtime, hardware type, number of cores, memory, datacenter PUE, and grid carbon intensity. Reports that a single ICON weather simulation generates ~2.5 tons CO₂/day.

**Key insight for us:** Their decomposition `Carbon = Runtime × Power × PUE × Carbon_Intensity` maps directly onto our measurement approach. We measure `Power` via nvidia-smi, `Runtime` via wall-clock, yielding `Energy = Σ(Power_i × Δt)` in Joules. Their framework validates our measurement methodology.

**Duplicity note:** Methodology overlaps with [10] Henderson but with a simpler, calculator-based approach vs. [10]'s tracking library.

---

### [10] Henderson et al. (2020) — Towards Systematic Reporting of Energy and Carbon Footprints
**Venue:** JMLR · **Relevance:** ★★★★

Proposes `experiment-impact-tracker`: a Python library with 13 standardized metrics for energy/carbon reporting. Surveys NeurIPS 2019: 0% of papers reported carbon impact; only 1% reported any energy metric. Provides leaderboard-style comparison framework. Metrics include: total power draw, GPU utilization, carbon equivalent, PUE-adjusted energy.

**Key insight for us:** Their 13-metric framework informs our logging design. We should report: total Joules, per-expert Joules, Joules-per-token, carbon equivalent (using grid intensity), and PUE-adjusted values. The 0%/1% reporting gap at NeurIPS motivates our work as a contribution to the Green AI reporting norm.

**Duplicity note:** Overlaps significantly with [9] Lannelongue in goals; differs in approach (Python tracker vs. web calculator). Both complement [7] Schwartz's call-to-action.

---

### [11] De Vogeleer et al. (2022) — PyJoules
**Venue:** Journal of Open Source Software · **Relevance:** ★★★

Python library for energy measurement using Intel RAPL (CPU/RAM) and NVIDIA APIs (GPU). Provides decorator-based and context-manager-based measurement with <5% overhead. Reports energy in Joules at the function/block level.

**Key insight for us:** PyJoules is a *complementary tool* to our pynvml-based energy_monitor.py. We use pynvml for real-time per-GPU power polling (feeding the cost vector `e_gpu` into the gating network); PyJoules could validate our measurements at the experiment level. The <5% overhead claim is important — our energy monitor must similarly not distort the measurements it's taking.

**Duplicity note:** Measurement tool like [12] nvidia-smi; differs in abstraction level (Python library vs. CLI utility). Both feed into [10] Henderson's reporting framework.

---

### [12] NVIDIA (2023) — nvidia-smi Documentation
**Venue:** NVIDIA Docs · **Relevance:** ★★★★

Documents the NVIDIA System Management Interface for GPU monitoring via NVML. Provides per-GPU power draw (Watts), temperature, SM/memory utilization, clock speeds, and ECC errors. `dmon` mode supports monitoring up to 16 GPUs simultaneously.

**Key insight for us:** nvidia-smi (via pynvml) is our **primary data source** for the energy cost vector `[e₁, e₂, …, eₙ]`. Critical caveat from recent research (Burtscher et al., 2023): on A100/H100, only ~25% of runtime is actually sampled for power — the rest is interpolated. Our 1-second averaging window in `energy_monitor.py` must account for this. We should validate pynvml readings against a known workload (torch.matmul at TDP) as specified in the M2 risk mitigations.

**Duplicity note:** Lower-level interface underlying [11] PyJoules' GPU measurement. Our pynvml usage is essentially a programmatic wrapper around nvidia-smi's power query.

---

## Cross-Cutting Paper

### [13] Hooker (2021) — The Hardware Lottery
**Venue:** Communications of the ACM · **Relevance:** ★★★

Position paper arguing that research ideas succeed or fail based on alignment with available hardware/software, not intrinsic merit. Historical examples: symbolic AI thrived on serial hardware; neural nets languished until GPU parallelism. Warns that specialized accelerators create lock-in to specific computational patterns.

**Key insight for us:** Our energy-aware routing is *hardware-coupled by design* — it reads GPU power states and adapts routing accordingly. This is both a strength (responds to real hardware conditions) and a risk (algorithm behavior depends on GPU heterogeneity). The Hardware Lottery perspective says: our method works well in heterogeneous multi-GPU settings but may be irrelevant on future hardware with uniform power profiles. We should discuss this as a limitation.

**Duplicity note:** Philosophical framing paper like [7] Schwartz; [7] focuses on compute cost, [13] on hardware-algorithm co-dependence. Both provide motivation but no experimental method.

---

## Summary: The Research Gap

| What exists | What's missing |
|---|---|
| MoE gating with load-balance losses [1-3] | Energy cost in the gating function |
| MoE expert analysis showing interchangeability [4-6] | Exploiting interchangeability for energy savings |
| Green AI manifestos and measurement tools [7-12] | Closing the loop: measurement → routing decision |
| Hardware-aware position papers [13] | Hardware-aware *training* algorithms |

**The gap our project fills:** No existing work feeds real-time, per-GPU energy costs back into the MoE routing decision. Papers [1-6] build the routing machinery; papers [7-12] build the measurement machinery; we connect them with `logits = x·W_g − λ·e_gpu`.
