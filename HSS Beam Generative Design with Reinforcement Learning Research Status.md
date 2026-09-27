# HSS Beam Generative Design with Reinforcement Learning
## Research Status and Proposed Final Study

### 1. Research Objective

The project investigates whether a **generative, amortized design approach** can automatically produce EC3-compliant HSS steel beam designs from structural demand.

The main inputs are:

- Beam span
- Uniformly distributed load

The design variables include:

- Beam depth and width
- Flange and web thickness
- Steel grade
- Rolled/welded section type

The design must satisfy relevant EC3 strength, stability, classification, and geometric/manufacturing constraints.

The main optimization objectives are:

- **Mass**
- **Cost**
- **CO₂**

---

## 2. Core Methodology

The study follows this general workflow:

```text
Structural demand
      ↓
Generative design model
      ↓
Candidate HSS beam
      ↓
Deterministic repair / refinement
      ↓
EC3 verification
      ↓
Feasible beam design
      ↓
Comparison with reference optimization
```

The research compares different ways of generating designs:

1. **PPO reinforcement learning**
2. **Classical optimization** such as GA and DE
3. **Simple amortized mapping**, including kNN

This allows us to study not only whether RL works, but **what value RL provides compared with conventional optimization and simpler learned approaches**.

---

## 3. Engineering and EC3 Verification

A substantial part of the work has focused on making the structural environment reliable before evaluating RL.

Completed checks include:

- EC3 implementation tests
- Independent hand verification of representative cases
- Verification of LTB behaviour
- Section classification and utilisation checks
- Geometry/manufacturing constraints
- Feasibility boundary analysis
- Ground-truth re-evaluation

Five representative EC3 hand calculations were independently checked with exact agreement for the tested utilisation/χLT calculations.

A previous LTB curve-selection issue for welded sections with high `h/b` was also identified and corrected.

The current implementation therefore provides a substantially stronger engineering basis than simply training an RL agent against an unverified reward function.

---

## 4. Ground Truth / Reference Designs

Reference designs are generated using conventional optimization procedures and are used to estimate how close generated designs are to high-quality solutions.

The current study uses:

- Multiple steel grades
- EC3 feasibility constraints
- Mass, cost and CO₂ objectives
- Multiple optimization restarts

The main evaluation set currently contains **142 structural contexts**.

A separate feasibility-envelope analysis tested **900 span/load combinations** and found approximately **0.7% structurally infeasible cases**, concentrated near the high-span/high-load boundary.

Ground-truth quality and dataset consistency are currently being audited because several generations of `pretrain_data` exist in the repository.

Therefore, these datasets will be consolidated into one authoritative final dataset before the final manuscript results are frozen.

---

## 5. Main Experimental Findings So Far

### 5.1 PPO can generate feasible designs

PPO successfully generates HSS beam designs satisfying the implemented feasibility constraints when combined with the current constraint-handling and deterministic refinement procedures.

This establishes the feasibility of the generative-design approach.

---

### 5.2 Deterministic post-processing is highly important

The current `scale + thin` procedure substantially improves generated designs.

Conceptually:

```text
PPO / GA / DE candidate
        ↓
scale toward a better feasible size
        ↓
try thinner dimensions
        ↓
EC3 verification
        ↓
improved feasible design
```

Importantly, the same procedure can be applied to different methods, allowing a fairer comparison.

---

### 5.3 Low-budget comparison

Corrected ExpA2 experiments compared PPO, GA and DE under low evaluation budgets.

At approximately 40 evaluations, the current corrected results indicate approximately:

| Method | Representation | Gap before repair | Gap after scale + thin |
|---|---|---:|---:|
| DE | Reparameterized | ~37% | **~23.7%** |
| GA | Reparameterized | ~41% | **~27.9%** |
| PPO | — | ~29% | **~7.5%*** |

`*` The ~7.5% PPO value is from a corrected single-run comparison. The current authoritative five-seed PPO result is approximately **9.05% mean cost gap**.

**Interpretation:** lower gap is better; 0% corresponds to the reference optimum.

These results suggest that PPO can produce competitive solutions with relatively little per-instance search, particularly after deterministic refinement.

However, this does **not** establish that PPO is universally superior to classical optimization.

---

## 6. High-Budget Optimization Finding

When classical optimizers are given substantially larger per-instance evaluation budgets, their solution quality improves considerably.

For example, at approximately 400 evaluations, current corrected results indicate:

- DE + reparameterization + scale+thin: approximately **4.9% mean gap**
- GA + reparameterization + scale+thin: approximately **6.8% mean gap**

This means classical optimization can eventually produce solutions better than the current five-seed PPO cost result (~9.05%).

This is an important finding rather than a weakness to hide.

It suggests that the main advantage being investigated is **amortized generation and computational efficiency**, rather than universal optimization superiority.

---

## 7. Amortization Finding

A major finding from ExpA3 is that the structural design problem appears to have a strong relationship between:

```text
Span + Load
      ↓
Near-optimal beam design
```

A simple k-nearest-neighbour (kNN) mapping achieved approximately **0.09% mean cost gap in-distribution** with the same deterministic refinement procedure.

This result is important because it shows that sophisticated RL is not necessarily required to exploit the demand-to-design structure of this particular low-dimensional problem.

Therefore, the research question has evolved from:

> "Can PPO beat classical optimization?"

toward:

> **"What does reinforcement learning contribute to amortized generative structural design compared with conventional optimization and simpler learned mappings?"**

---

## 8. OOD Generalization

The study also investigates extrapolation outside the training demand range.

### Span extrapolation

Training range:

```text
6–15 m
```

Testing range:

```text
16–22 m
```

Current indicative results:

- kNN: approximately **0.99% gap**
- PPO: approximately **12.1% gap**

### Load extrapolation

Training range:

```text
20–140 kN/m
```

Testing range:

```text
150–260 kN/m
```

Current indicative results:

- kNN: approximately **1.61% gap**
- PPO: approximately **12.5% gap**

However, kNN feasibility deteriorates more substantially under the tested extrapolation, whereas PPO retains better feasibility behaviour.

These results therefore indicate a **quality–feasibility trade-off**, rather than a simple winner.

The OOD statistics still require final verification before being treated as manuscript-authoritative.

---

## 9. Multi-Objective Finding

The current experiments demonstrate that optimizing one objective does not automatically optimize the others.

For the five-seed cost-trained PPO evaluation, current combined-gap results are approximately:

| Objective | Mean gap |
|---|---:|
| Cost | **9.05%** |
| Mass | **28.12%** |
| CO₂ | **48.04%** |

This means we should **not** claim that a cost-trained PPO simultaneously produces near-optimal mass, cost and CO₂ designs.

The scientifically defensible interpretation is that objective-specific optimization matters.

---

## 10. Current Research Contribution

The potential contribution is therefore broader than simply applying PPO to HSS beams.

The study aims to provide a controlled investigation of:

1. **EC3-compliant generative structural design using RL**
2. **Amortized design generation from structural demand**
3. **Comparison with classical per-instance optimization**
4. **Comparison with a simple non-RL learned mapping**
5. **Effect of design-space representation and deterministic repair**
6. **Low-budget versus high-budget optimization behaviour**
7. **Generalization to unseen structural demands**
8. **Feasibility versus objective-quality trade-offs**
9. **Objective-specific behaviour for mass, cost and CO₂**

A potentially important community-level contribution is therefore an evidence-based analysis of **when RL is useful for generative structural design, rather than assuming that RL is inherently superior.**

---

## 11. What We Can Safely Claim Now

Based on the current evidence, the following claims are supportable subject to final verification:

- An EC3-based HSS beam generative-design environment has been implemented and independently checked.
- PPO can generate feasible HSS beam designs.
- Deterministic post-processing substantially improves generated designs.
- PPO can be competitive with tested GA/DE methods at low per-instance evaluation budgets.
- Higher-budget GA/DE optimization can achieve lower absolute gaps.
- The problem exhibits strong amortization structure.
- A simple kNN mapping can perform extremely well in-distribution.
- PPO and kNN show different behaviour under OOD extrapolation, particularly regarding objective quality and feasibility.
- Optimizing cost does not automatically optimize mass and CO₂.

We should **not** claim:

- PPO is universally better than GA/DE.
- RL is necessary for generative structural design.
- PPO produces near-optimal mass, cost and CO₂ simultaneously.
- The current results prove global optimality.
- OOD superiority of one method without considering both quality and feasibility.

---

## 12. Current Limitations

Important remaining limitations include:

- Multiple generations of ground-truth datasets currently exist and require final consolidation.
- Some historical experiments use different reward/configuration choices.
- Not every mass/cost/CO₂ experiment has yet been confirmed as part of one consistent final pipeline.
- The current rolled-section representation is procedural rather than a complete manufacturer catalogue.
- The current structural loading formulation focuses on the selected beam/loading case.
- Cost and CO₂ depend on assumed coefficients and require sensitivity analysis if they are central conclusions.
- The continuous PPO environment's treatment of steel grade should be clearly documented and justified.
- Third-party structural-software verification has not been established.
- Full multi-objective/Pareto optimization has not been implemented.

These limitations will be addressed where necessary for the final research question rather than adding experiments solely to increase the number of results.

---

## 13. Proposed Final Research Workflow

Before finalizing the manuscript:

```text
1. Audit all existing experiments
        ↓
2. Identify authoritative vs obsolete results
        ↓
3. Freeze one final ground-truth dataset
        ↓
4. Freeze the final experimental protocol
        ↓
5. Re-run only experiments that are necessary
        ↓
6. Complete mass / cost / CO₂ evaluation
        ↓
7. Complete final PPO multi-seed evaluation
        ↓
8. Complete GA / DE / kNN comparisons
        ↓
9. Verify OOD and statistical results
        ↓
10. Freeze final results
        ↓
11. Rewrite manuscript around the evidence
        ↓
12. Release reproducible code/data/results
```

---

## 14. Intended Final Paper Message

The intended message is **not**:

> "RL beats traditional optimization."

Instead:

> **This study investigates reinforcement learning as an amortized generative-design mechanism for EC3-compliant HSS beams and systematically examines its performance against conventional optimization and simpler learned mappings. The results reveal both the opportunities and limitations of RL, including the effects of computational budget, design representation, deterministic refinement, objective choice, and extrapolation beyond the training domain.**

This provides a more balanced and potentially more meaningful contribution to the structural engineering and generative-design community.