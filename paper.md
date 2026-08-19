# Generative Design of High-Strength Steel (HSS) Beams via Proximal Policy Optimization PPO: A Demand-Driven, EC3-Compliant Reinforcement Learning Framework

---

## Abstract

This project develops and validates a reinforcement-learning (RL) framework for automated, code-compliant structural design of Eurocode 3 (EC3) steel beams, with a specific focus on **demand-driven material grade selection** across the full range of structural steel grades (S355 to S690). A Proximal Policy Optimization (PPO) agent is trained to iteratively propose section geometry (depth, width, flange and web thickness), steel grade, and fabrication type (rolled or welded) for randomly sampled span/load scenarios, learning — without any hand-labeled design examples — to reserve high-strength steel (HSS) for spans and loads that genuinely justify its cost premium, while defaulting to conventional steel (S355/S460) elsewhere. The final trained agent achieves **97.0% feasibility** across 200 randomly sampled design scenarios and **96.9% coverage (62/64 cells) on a structured 8×8 span×load validation grid**, with utilization concentrated tightly around the EC3-optimal target (mean 0.954, std 0.019) and demand-appropriate use of HSS grades (S620+S690 selected in 40.2% of feasible designs, concentrated at the longest spans and heaviest loads). This report documents the environment design, the iterative experimental process used to reach this result (including several methodologically important negative results), the final validated performance, and a proposed direction for extending this work toward multi-element structural systems using multi-agent and graph-based RL.

---

## 1. Introduction

### 1.1 Motivation

Structural steel design is a constrained, multi-objective optimization problem: a beam must satisfy moment capacity, shear capacity, lateral-torsional buckling (LTB), deflection, and section-classification requirements under Eurocode 3 (EC3), while minimizing mass, material cost, and (increasingly) embodied carbon. The design space includes not only continuous geometric parameters but also a **discrete material choice** — the steel grade — whose cost and carbon footprint scale non-trivially with strength. Classical optimization methods (genetic algorithms, gradient-based search, simulated annealing) can solve individual instances of this problem but must be re-run from scratch for every new span/load combination, and none of them learn a *transferable design policy*.

### 1.2 Related work

Jeong & Jo (2021), *"Deep reinforcement learning for automated design of reinforced concrete structures,"* published in *Computer-Aided Civil and Infrastructure Engineering*, is the most directly relevant prior work. That study trained a Deep Deterministic Policy Gradient (DDPG) agent with a convolutional actor-critic network to design ACI 318-compliant reinforced concrete (RC) beams, using a five-step episode structure, custom reward functions penalizing code violations and rewarding cost-effective designs, and validated the trained agent against 100 randomly generated cases and two comparative case studies (a simple beam and a three-bay building frame). Notably, that paper's conclusion **explicitly names PPO as unexplored future work** for this problem class.

### 1.3 Contributions and uniqueness of this project

This project extends the Jeong & Jo line of work in three specific, verifiable respects:

1. **Algorithm.** PPO is used in place of DDPG — a gap explicitly identified in the cited paper's own conclusion. PPO's clipped surrogate objective and on-policy stability characteristics are, a priori, well suited to a reward landscape with several interacting objective terms (utilization, economy, CO₂, grade-appropriateness, feasibility), which this project's own experimental history (Section 3) shows to be sensitive to training dynamics.

2. **Design code and structural complexity.** EC3 steel design introduces mechanics with no analogue in the RC case: lateral-torsional buckling capacity (`Mcr`, `χ_LT`) computed from full torsional/warping section properties, grade-dependent section classification via `epsilon = sqrt(235/f_y)` (higher-grade steel has *stricter* compactness limits at identical proportions — a real, non-obvious EC3 effect that this project's own investigation surfaced and had to distinguish from implementation bugs; see Section 3.4), and shear-moment interaction affecting the effective moment resistance.

3. **Learned material selection.** The RC case study has a single, fixed material specification; concrete strength is not an agent decision. This project's action space includes a **six-way steel-grade selection** (S355, S460, S500, S550, S620, S690) as a first-class, per-step agent decision, jointly optimized against geometry. This is the paper's core novel claim: that a PPO agent can learn, purely from reward signal and without labeled examples, the same qualitative judgment a structural engineer applies — that HSS grades are only economically and structurally justified once conventional steel would require excessive mass or section depth.

### 1.4 Scope of this report

This report covers: the environment and training design (Section 2), the experimental methodology used to reach the final result, including negative results and diagnostic findings that are methodologically significant in their own right (Section 3), the final validated results (Section 4), a discussion of what was and was not solved (Section 5), reproducibility artifacts (Section 6), and a proposed research direction extending this framework to full structural systems via multi-agent and graph-based RL (Section 7).

---

## 2. Methodology

### 2.1 Problem formulation as a Markov Decision Process

At the start of each episode, a span (6–15 m), distributed load (20–210 kN/m, storey-scaled), and storey count (1–70) are sampled uniformly. The agent then has up to 40 steps to iteratively refine a design, observing the current state and selecting a continuous action at each step.

**State (25-dimensional):** span, load, storey count (normalized); current section geometry (`h`, `b`, `tf`, `tw`) and their ratio; current steel grade and section type; current utilization, moment ratio, mass, cost, CO₂, LTB reduction factor (`χ_LT`); flange and web slenderness ratios; one-hot EC3 section classification; step-over-step utilization and mass deltas; and a normalized design-moment feature (`Med_norm = w·L²/8`, direct structural demand signal).

**Action (6-dimensional, continuous, tanh-bounded):** four geometry deltas (Δh, Δb, Δtf, Δtw, applied with a cosine-annealed step-size schedule that shrinks over the episode); one grade-selection dimension, mapped to a probability distribution over the six discrete grades via a temperature-scaled softmax over grade "centres" in action space (rather than direct discretization), argmax-snapped to the selected grade — this design choice, and the reasoning behind it, is discussed in Section 3.2; and one section-type (rolled/welded) selector.

**Reward:** a weighted sum of (a) a Gaussian utilization-targeting term peaking at util = 0.96, penalized more steeply above 1.05 (infeasible overstress); (b) an economy term penalizing mass and cost; (c) a demand-conditioned CO₂ lifecycle bonus (rewarding lower embodied carbon per unit of moment capacity, gated to only apply near-feasible states); (d) a demand-driven HSS bonus (rewarding grade ≥ S500 selection specifically when structural demand and utilization jointly justify it — this is the term most directly responsible for the paper's core "demand-driven grade transition" claim); (e) an underutilization penalty (quadratic below util = 0.90); (f) feasibility penalties for moment/shear/class violations and unrealistic geometry ratios; (g) a mass-improvement shaping term; and (h) a geometry-novelty bonus discouraging premature convergence to a single design.

**Termination:** an episode terminates early after three consecutive steps within the feasible band (0.90 ≤ util ≤ 1.05, EC3 class ≤ 3), or is truncated at 40 steps.

### 2.2 Training configuration

PPO (Stable-Baselines3 implementation), 8 parallel environments via `SubprocVecEnv`, observation and reward normalization via `VecNormalize`, policy/value network architecture `[256, 256, 128]` (separate heads), entropy coefficient 0.03, learning rate 3×10⁻⁴, clip range 0.15, evaluated every 25,000 timesteps against a fixed-seed held-out environment. The final reported model was trained for **1,000,000 timesteps**.

### 2.3 Validation methodology

Three complementary evaluation tools were used, each addressing a different failure mode:

- **`validate.py`** — 200 randomly sampled episodes plus a structured 8×8 span×load grid (spans 6–15 m, loads 20–140 kN/m), reporting feasibility rate, utilization/mass/cost/CO₂ statistics, grade distribution, and per-grade efficiency. This is the primary aggregate-performance instrument.
- **`diagnostic_grade_comparison.py`** — freezes the policy's chosen geometry at five representative hard scenarios and sweeps all six grades against it, cross-checked against a brute-force minimum-mass EC3 geometry search. This isolates whether a given design's grade choice, independent of geometry, is sound.
- **`diagnostic_stability_trace.py`** — logs every single step of a rollout (raw action values per dimension, resulting geometry, grade, utilization, and whether that step is in the feasible band), used to distinguish "never converges" from "converges then destabilizes" from "oscillates" — a distinction that proved essential (Section 3.5).

An independent **brute-force EC3-optimal ground-truth dataset** (`pretrain_data/ec3_optimal_designs.csv`, 1,174 feasible span/load/grade/type contexts, generated by exhaustive grid search over geometry at each context) was built specifically to validate policy behavior against ground truth rather than relative changes in a validation metric — this dataset is used both in Section 4.4 and is proposed as the basis for a reference-design comparison analogous to Jeong & Jo's Section 5.6 (see Section 8).

---

## 3. Experimental process (summary)

The final architecture is the product of an extensive iterative process (full detail in `docs/EXPERIMENT_LOG.md`), which is itself methodologically relevant to report rather than omit, since it surfaces several general lessons about validating RL policies for engineering design.

### 3.1 Early iteration (exp46–exp53): reward and sampling instability

Initial experiments established core EC3 mechanics and reward shaping (exp46–exp50), reaching 97.5% feasibility at exp48. A subsequent attempt to give the grade action *memory* (a persistent, incrementally-adjusted grade rather than a fresh choice each step) coupled grade selection to geometry in an unintended way: the agent learned "start at S690, hold it," collapsing feasibility to 44–49% at exp51–53. This was diagnosed as a genuine grade-geometry coupling artifact of the action design, not a reward problem.

### 3.2 exp54: the architecture reported as final

Replacing the persistent grade action with a **softmax-snapped, memoryless** grade selection (re-decided fresh every step from a temperature-scaled Gaussian similarity to six grade "centres" in action space) restored feasibility to 96.5% while correctly reintroducing demand-driven grade diversity. **This architecture, trained for 1,000,000 timesteps rather than the original 500,000, is the model reported as final in this project** (Section 4).

### 3.3 A validation-methodology bug with a material effect on conclusions

During diagnostic development, a bug was identified in the evaluation tooling: Stable-Baselines3's `VecEnv` contract automatically resets an environment *inside* any `step()` call that returns `done=True`, before control returns to the caller. Code that read a policy's terminal design via live environment attributes immediately after such a call was, without any error or warning, silently reading the *auto-reset's fresh random draw* instead of the policy's actual output. Because no randomness is consumed anywhere in the environment's step dynamics (only at `reset()`), and every diagnostic scenario re-seeded an identical RNG state, this produced bit-identical "converged geometry" across every tested scenario regardless of span or load — a signature indistinguishable, on cursory inspection, from total policy collapse. This was confirmed directly by comparing the environment's `info` dict (populated correctly, before any reset) against the same environment's live attributes read immediately afterward, which differed. **This bug materially affected which checkpoints appeared to be "working" during development, and its correction is reported here as a methodological finding relevant to any RL-for-design study relying on custom diagnostic tooling.**

### 3.4 A genuine EC3 physics effect initially mistaken for noise

During construction of the brute-force EC3-optimal dataset, apparent "mass inversions" (higher-grade steel requiring *more* mass than a lower grade at the same demand) were investigated as possible dataset-generation bugs. Two genuine, EC3-consistent physical effects were identified as the actual cause: (1) deflection depends only on Young's modulus, identical across all steel grades, so deflection-governed designs see *zero* mass benefit from higher grade; (2) `epsilon = sqrt(235/f_y)` decreases with increasing grade, making section-classification compactness limits *stricter* for HSS at identical proportions, occasionally forcing a higher-grade section to be stockier (heavier) than a lower-grade one just to remain compact. Both effects are correctly reproduced by the environment's EC3 mechanics and are retained as genuine physics rather than "fixed" as if they were errors — a useful example of the general principle that anomalous RL behavior in a physically-grounded environment should be checked against domain physics before being treated as a training artifact.

### 3.5 exp57–exp62: a residual-oscillation investigation, and its outcome

Clean diagnostics on exp54 (after the fixes in Section 3.3) showed the policy correctly conditions on span/load and reaches good designs in the great majority of tested scenarios, but in a narrow region of the demand space (span ≥ 11 m, load ≥ 100 kN/m) the policy sometimes finds a genuinely feasible design without holding it for the three consecutive steps required for early termination. Six subsequent experiments attempted targeted fixes — a reward term penalizing post-success grade changes (exp57), grade-selection hysteresis (exp58), geometry-action damping (exp59), grade tier-adjacency restriction (exp60), physics-informed flange/web-thickness observation hints fitted against the ground-truth dataset (exp61), and a relaxed termination criterion (exp62). Each fix suppressed the specific symptom it targeted; in three cases (exp58, exp59, exp61) it did so without a broader regression, but in three cases (exp57, exp60's broader validation, exp62) it produced a **net regression in overall feasibility** when checked against the full validation grid rather than only the two scenarios that motivated the fix — a direct illustration of the risk of over-fitting an RL policy modification to a small number of hand-picked diagnostic cases.

### 3.6 The decisive ablation: training budget versus architectural engineering

Before adopting exp61 (which stacked five changes across exp57–61 and was trained for 1,000,000 timesteps) as final, the confound of unequal training budgets was identified: the original exp54 baseline had only been trained for 500,000 timesteps, half of exp61's. **The unmodified exp54 environment was retrained for the same 1,000,000-timestep budget ("exp54b") as a controlled comparison.** The result is reported in full in Section 4; in summary, exp54b matched or exceeded exp61 on every headline metric, despite containing none of the five architectural changes. This is treated as the decisive finding of the experimental process: **additional training budget alone, on the original, unmodified environment, outperformed five rounds of targeted architectural intervention.**

---

## 4. Results

### 4.1 Final model performance (exp54b: exp54 architecture, 1,000,000 timesteps)

| Metric | Value |
|---|---|
| Feasibility (200 random episodes) | **97.0%** (194/200) |
| In target band [0.90, 1.05] among feasible | 100.0% |
| Grid coverage (8×8 span×load) | **96.9%** (62/64 cells feasible) |
| Mean utilization (feasible episodes) | 0.954 (σ = 0.019) |
| Mean section class | Class 1 in 98.5% of feasible episodes |
| HSS usage (S620 + S690) | 40.2% of feasible episodes |
| Grade coverage | All six grades represented (S355: 6.2%, S460: 14.9%, S500: 22.2%, S550: 16.5%, S620: 15.5%, S690: 24.7%) |

### 4.2 Demand-driven grade selection

The grid-validation grade-transition table shows the intended qualitative pattern: at short spans / light loads (e.g., 6 m, 20–80 kN/m), the policy consistently selects S355; at increasing demand, selection shifts through S460, S500, S550 toward S620/S690 at the longest spans and heaviest loads (e.g., 10–11 m at 30–140 kN/m, 15 m at 80–140 kN/m). This is the core qualitative claim of the paper — that the agent has learned the same *reserve HSS for genuinely demanding cases* heuristic a structural engineer applies — and it is directly observable in this table, not merely inferred from aggregate statistics.

### 4.3 Ablation: environment architecture versus training budget

| Configuration | Timesteps | Feasibility | Infeasible grid cells | HSS usage |
|---|---|---|---|---|
| exp54 (original) | 500,000 | 96.5% | 3 / 64 | 30.0% |
| exp61 (5 architectural changes) | 1,000,000 | 94.5% | 5 / 64 | 30.1% |
| **exp54b (unmodified exp54, extended training)** | **1,000,000** | **97.0%** | **2 / 64** | **40.2%** |

This table is the paper's central ablation finding and should be reported prominently: it demonstrates that the architectural interventions attempted in exp57–61, despite being individually well-motivated and evidence-driven, were not necessary to reach the best result obtained in this project — and in exp61's case, on balance, cost more (in feasibility and grid coverage) than they gained (a validated but narrower tf/tw calibration improvement, see 4.4).

### 4.4 A validated secondary finding: physics-informed observation features for flange/web thickness

Independent of the ablation conclusion above, one specific finding from the exp57–61 line is retained as a validated, dataset-verified contribution: the trained policy (across multiple checkpoints, not only exp60) was found to drive flange and web thickness toward their design-limit ceiling in certain scenarios where the brute-force EC3-optimal dataset shows this is unnecessary — checked broadly across 99 span/load contexts in the ground-truth dataset, only 1% ever require web thickness ≥ 20 mm even in the *thinnest* feasible design at that context, meaning the ceiling-pinning behavior is a general miscalibration rather than a scenario-specific artifact. A quadratic regression fit against the same ground-truth dataset explains true optimal flange/web thickness far better than the crude heuristic previously used only at episode initialization (R² = 0.836 and 0.766 respectively, versus R² = −0.295 and −2.348 for the prior heuristic — i.e., the prior heuristic was worse than simply predicting the mean). This finding is independent of the training-budget ablation and worth reporting as a validated, if secondary, contribution — for instance as an appendix or a note on future architectural refinement.

### 4.5 A confirmed, disclosed limitation

In the extreme demand corner of the sampled design space (span 11–15 m, load ≥ 100 kN/m), the final model (exp54b) — like every checkpoint tested during this project's entire experimental history, including those purpose-built to address it — sometimes reaches a genuinely feasible design without holding it for three consecutive steps, running the full 40-step episode instead of terminating early. This was directly confirmed via per-step trajectory tracing on exp54b: at a representative hard scenario (13 m span, 100 kN/m load), the policy touches the feasible band at 11 separate points across a 40-step trajectory without ever holding three in a row. Grid validation independently confirms the *best design found* in this region is feasible, so this limitation affects training-time efficiency in a narrow, underrepresented region of the demand space, not the reported feasibility numbers or the quality of the best design found. Six dedicated experiments (Section 3.5) support the conclusion that this is most likely PPO advantage-estimate noise specific to an underrepresented region of the sampled demand distribution, rather than a defect in the environment's reward or action design correctable by further environment engineering.

---

## 5. Discussion

### 5.1 What this project solved

- A working, validated PPO-based RL environment for EC3-compliant steel beam design, extending an RC/DDPG precedent to a materially more complex design-code and action space.
- Demonstrated, dataset-verifiable evidence that the trained agent learns demand-driven material grade selection without labeled training examples — a nontrivial result given that grade choice interacts with geometry, cost, and CO₂ in reward terms that could plausibly have produced a degenerate always-cheapest or always-strongest policy instead.
- A validated, ground-truth-checked correction to flange/web-thickness miscalibration, independently useful regardless of which overall architecture is adopted.
- A rigorously demonstrated ablation separating the effect of training budget from architectural intervention — a result with methodological value beyond this specific project, given how often RL-for-design papers report a single training run per configuration without this kind of control.
- Identification and correction of a subtle, otherwise-invisible validation-tooling bug (Section 3.3) that materially affected which checkpoints appeared to work during development — worth reporting as a caution for similar future work.

### 5.2 What remains open

- The residual oscillation in the extreme demand corner (Section 4.5) is disclosed, not resolved. It is not believed to be resolvable by further environment engineering given the evidence gathered, but this remains a hypothesis, not a proof; a larger and more systematic sweep of PPO hyperparameters (batch size, `n_steps`, entropy schedule) specifically targeting this region, or a curriculum that increases episode exposure to it without the sampling-bias side effects observed in exp56, remains untested.
- All results in this report derive from a **single training run per configuration** (with the sole exception of the exp54/exp54b timestep-budget comparison). Given PPO's well-documented run-to-run variance, multi-seed replication (as done in the Jeong & Jo precedent, which averaged five trained agents per configuration) would materially strengthen any claims made in a publication and is the most important immediate next step before submission.
- The environment currently designs a single, simply-supported beam in isolation. Real structural design involves interacting members (continuous beams, frames, load paths through connected elements) — the natural next extension, addressed in Section 7.

---

## 6. Reproducibility and provided artifacts

All code, the environment, training script, three-tool validation suite, ground-truth EC3 dataset generation pipeline, and full experiment log are provided in the accompanying `hss_beam_rl/` package. `docs/EXPERIMENT_LOG.md` in that package contains the complete, detailed version of Section 3 above, including per-experiment configuration deltas, formulas, and citation-ready numeric evidence for the ablation and the tf/tw finding. Trained model checkpoints (`best_model.zip`, `vecnormalize.pkl`) are not included in the package, since they are specific to the training machine, but the package is structured so that training from scratch reproduces the reported configuration exactly.

---

## 7. Future work: extending to multi-element structural systems

The present environment designs one beam at a time, in isolation, for a directly-specified span and load. Real structural design — and the natural next step for this line of work, mirroring the progression in the cited RC precedent from a single beam to a three-bay frame case study — requires reasoning about **multiple interacting structural elements** simultaneously: continuous beams sharing supports, frames where beam and column design are coupled, and load paths that redistribute when one member's design changes. Two complementary extensions are proposed.

### 7.1 Multi-agent reinforcement learning (MARL)

**Motivation.** A single monolithic agent designing an entire frame at once faces an action space that grows with the number of members, and a reward signal that must somehow attribute credit for a global outcome (e.g., total structure cost) back to individual member decisions — a classical multi-agent credit-assignment problem. A more natural formulation assigns **one agent per structural member** (or per member type — beams, columns), each observing its own local demand plus shared global state (e.g., total structure cost/CO₂, or forces transferred from adjacent members), and acting on its own design variables.

**Proposed approach.** A centralized-training, decentralized-execution (CTDE) architecture — e.g., QMIX-style value decomposition or a multi-agent PPO variant (MAPPO) — is a natural fit given this project's existing PPO infrastructure. Each member-agent's local reward would combine its own EC3 feasibility/economy terms (as in the current single-beam reward) with a shared global term reflecting total structure performance, encouraging cooperative rather than purely locally-optimal designs. The existing single-beam environment's EC3 mechanics (`_ec3_analysis`, `_calculate_cost_co2`) are directly reusable as the per-agent local reward computation; the new work is in defining the inter-agent state-sharing protocol and the frame-level load-distribution mechanics connecting member design decisions to each other's effective demand.

**Expected challenges, stated honestly.** Non-stationarity (each agent's environment changes as other agents update their policies), credit assignment for the shared global term, and a substantially larger validation burden (per-member diagnostics, not just per-scenario) are all real risks that should be scoped explicitly in a follow-up proposal rather than assumed away.

### 7.2 Graph neural network (GNN) representations

**Motivation.** A structural frame is naturally a graph: members are edges, joints/supports are nodes, and load paths follow graph connectivity. The current environment's fixed-size, hand-engineered observation vector (25 features for one beam) does not generalize to frames with a varying number of members and connectivity patterns — a new frame topology would require a new observation design. A graph neural network (GNN) encoder, by contrast, can represent an arbitrary frame topology as a graph (node features: joint location, support conditions, applied loads; edge features: current member geometry, grade, section forces) and produce a fixed-size latent representation usable by a PPO policy/value head regardless of the frame's size or shape.

**Proposed approach.** Replace the current environment's flat observation vector with a graph representation of the structure (initially: a continuous multi-span beam, as an intermediate step before full 2D/3D frames, mirroring the RC precedent's own progression from simple beam to frame case study). A GNN (e.g., a graph attention network, given the physical intuition that load-path influence between members is distance- and connectivity-dependent) would encode this graph into the state representation consumed by the PPO actor-critic, with actions still per-member but now informed by a topology-aware, shared encoder — enabling, in principle, **generalization across frame topologies without retraining**, a capability neither this project's current environment nor the cited RC precedent's frame case study (which retrains a fresh policy per topology) currently has.

**Relationship to the multi-agent extension.** These two directions are complementary, not competing: a MARL formulation with per-member agents naturally benefits from a GNN-based shared or partially-shared state encoder, since it directly supplies each agent with a structurally-grounded representation of how its member relates to the rest of the system. A combined **GNN-encoded, multi-agent PPO framework for full-frame EC3-compliant generative design** is the recommended long-term direction for this research line.

### 7.3 Suggested near-term sequencing

Given the scope above, the following order is suggested rather than attempting all of it simultaneously: (1) multi-seed replication of the current single-beam result (addresses the most important open gap identified in Section 5.2, and is required regardless of which extension is pursued next); (2) a two- or three-span continuous beam as the first multi-element extension, using a hand-designed (non-GNN) shared state to establish whether the MARL formulation itself is tractable before adding representation-learning complexity; (3) GNN state representation, introduced once the multi-agent formulation is validated on the continuous-beam case; (4) full 2D frame, replicating the cited RC precedent's own final case study but with the grade-selection and GNN/MARL extensions this project contributes.

---

## 8. Recommended next steps before submission

1. **Multi-seed replication** of the exp54b result (minimum 3–5 seeds), reporting mean ± standard deviation for all headline metrics in Section 4.1, following the precedent set by the cited RC paper.
2. **Reference-design comparison** against the brute-force EC3-optimal dataset already built (`pretrain_data/ec3_optimal_designs.csv`) at a representative set of span/load points, analogous to the cited paper's Section 5.6 comparison against textbook-designed examples — this project's version can use a much larger N (the full dataset) rather than three hand-picked cases.
3. **A short, explicit parametric/ablation study section** built directly from Section 4.3 of this report and the exp57–62 history in `docs/EXPERIMENT_LOG.md`, mirroring the cited paper's Section 5.5 (step-size and reward-function sensitivity study) but with this project's own natural knobs (training budget vs. architectural intervention; grade-curriculum ratio; entropy coefficient).
4. **Manuscript drafting**, following the section structure of the cited RC precedent (Introduction → Environment → Algorithm → Training/Validation → Parametric study → Reference-design comparison → Conclusion), explicitly citing Jeong & Jo (2021) in Related Work as the closest prior study and naming PPO, EC3/HSS grade selection, and the training-budget-vs-architecture ablation as this project's contributions beyond it.

---

## Reference

Jeong, J.-H., & Jo, H. (2021). Deep reinforcement learning for automated design of reinforced concrete structures. *Computer-Aided Civil and Infrastructure Engineering*, 36(12), 1508–1529. https://doi.org/10.1111/mice.12773
