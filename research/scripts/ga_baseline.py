"""
research/scripts/ga_baseline.py
================================================================
Genetic-algorithm baseline for HSS beam sizing, using the SAME EC3
mechanics and cost model as the RL environment (imported directly from
research/envs/hss_env.py -- not reimplemented, so there is zero risk of
the baseline silently using different physics than the RL arms).

WHY THIS BASELINE, SPECIFICALLY
---------------------------------
This mirrors Jeong & Jo (2021, the RC-beam DRL paper this project extends)
Section 5.6-5.7, which benchmarked their DDPG agent against a GA and a
Big Bang-Big Crunch optimizer. The point of this comparison is NOT "does
GA beat RL" (a per-instance GA re-optimized from scratch will often match
or beat a fixed trained policy on solution quality, precisely because it
gets to search fresh every time) -- the point is to report the trade-off
honestly: GA's per-query cost (population x generations x EC3 evaluations,
every single time a new span/load context arrives) versus RL's amortized
cost (~0.1s policy forward pass at inference, after a one-time training
cost). This is the paper's clearest, most defensible "why RL" argument,
and it can only be made by actually running both and reporting real
wall-clock numbers side by side.

ALGORITHM
----------
Real-valued GA (not binary-encoded) operating directly on the six design
variables (h, b, tf, tw, fy, section_type), for a FIXED (span, load,
storey) context per call -- i.e. one independent optimization run per
test case, exactly matching how a practising engineer would use a GA
tool: reformulate and re-solve for every new demand.

    Genome: [h, b, tf, tw, fy_index, section_type_index] (mixed real/int)
    Fitness: -economy(design) if feasible, else -(BIG + violation_magnitude)
             (same constraint-violation functions as the Lagrangian arm,
             for a fully apples-to-apples feasibility/economy definition)
    Selection: tournament (k=3)
    Crossover: BLX-alpha for continuous genes, uniform for discrete genes
    Mutation: Gaussian for continuous genes, random-reset for discrete genes
    Elitism: top 2 preserved each generation

USAGE
------
    from research.scripts.ga_baseline import ga_design
    result = ga_design(span_mm=8000, load_kNm=60, storey=20,
                        economy_metric="cost", pop_size=60, n_generations=80, seed=0)
    print(result)  # dict with best design, economy, feasibility, wall-clock time, n_evaluations
================================================================
"""

import time
import numpy as np

from research.envs.hss_env import HSSBeamEnv

GRADES = np.array([355, 460, 500, 550, 620, 690], dtype=np.float64)
SECTION_TYPES = ["rolled", "welded"]
BOUNDS = dict(h=(250.0, 750.0), b=(120.0, 300.0), tf=(8.0, 35.0), tw=(6.0, 25.0))


def _make_probe_env(economy_metric):
    # A single HSSBeamEnv instance is reused purely as a stateless physics
    # calculator: we directly set its (h,b,tf,tw,fy,section_type,span,load,
    # storey) attributes and call _ec3_analysis()/_calculate_cost_co2()/
    # _constraint_violations()/_economy() -- the exact same functions the
    # RL arms are scored with -- without going through step()/reset()'s
    # RL-specific bookkeeping (curriculum sampling, episode counters, etc).
    env = HSSBeamEnv(reward_mode="lagrangian", economy_metric=economy_metric)
    return env


def _evaluate(genome, env, span_mm, load_kNm, storey):
    h, b, tf, tw, grade_idx_f, type_idx_f = genome
    env.h, env.b, env.tf, env.tw = float(h), float(b), float(tf), float(tw)
    env.fy = float(GRADES[int(np.clip(round(grade_idx_f), 0, len(GRADES) - 1))])
    env.section_type = SECTION_TYPES[int(np.clip(round(type_idx_f), 0, 1))]
    env.span, env.load, env.storey = float(span_mm), float(load_kNm), int(storey)

    util, mass, penalty, class_loss, chi_lt, dbg = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    violations = env._constraint_violations(util, class_loss, penalty)
    feasible = all(v <= 1e-3 for v in violations.values())
    economy = env._economy(mass, cost, co2)

    if feasible:
        fitness = -economy
    else:
        fitness = -(50.0 + 20.0 * violations["g1_util"]
                     + 10.0 * violations["g2_class"] + 5.0 * violations["g3_geom"])
    return fitness, dict(util=util, mass=mass, cost=cost, co2=co2, feasible=feasible,
                          section_class=dbg["section_class"], h=env.h, b=env.b, tf=env.tf,
                          tw=env.tw, fy=env.fy, section_type=env.section_type)


def _random_genome(rng):
    return np.array([
        rng.uniform(*BOUNDS["h"]), rng.uniform(*BOUNDS["b"]),
        rng.uniform(*BOUNDS["tf"]), rng.uniform(*BOUNDS["tw"]),
        rng.integers(0, len(GRADES)), rng.integers(0, 2),
    ], dtype=np.float64)


def _clip_genome(g):
    g[0] = np.clip(g[0], *BOUNDS["h"]); g[1] = np.clip(g[1], *BOUNDS["b"])
    g[2] = np.clip(g[2], *BOUNDS["tf"]); g[3] = np.clip(g[3], *BOUNDS["tw"])
    g[4] = np.clip(round(g[4]), 0, len(GRADES) - 1)
    g[5] = np.clip(round(g[5]), 0, 1)
    return g


def ga_design(span_mm, load_kNm, storey, economy_metric="cost",
              pop_size=60, n_generations=80, seed=0, tournament_k=3,
              crossover_alpha=0.3, mutation_rate=0.25, mutation_sigma_frac=0.10, elitism=2):
    rng = np.random.default_rng(seed)
    env = _make_probe_env(economy_metric)

    pop = [_random_genome(rng) for _ in range(pop_size)]
    fitness_evals = 0
    t0 = time.time()
    best_genome, best_fitness, best_meta = None, -np.inf, None

    for gen in range(n_generations):
        scored = []
        for g in pop:
            fit, meta = _evaluate(g, env, span_mm, load_kNm, storey)
            fitness_evals += 1
            scored.append((fit, g, meta))
            if fit > best_fitness:
                best_fitness, best_genome, best_meta = fit, g.copy(), meta

        scored.sort(key=lambda t: t[0], reverse=True)
        new_pop = [scored[i][1].copy() for i in range(elitism)]

        def tournament():
            idxs = rng.integers(0, pop_size, size=tournament_k)
            best_idx = max(idxs, key=lambda i: scored[i][0])
            return scored[best_idx][1]

        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            child = np.empty(6)
            for i in range(4):  # continuous genes: BLX-alpha
                lo, hi = min(p1[i], p2[i]), max(p1[i], p2[i])
                span = hi - lo
                child[i] = rng.uniform(lo - crossover_alpha * span, hi + crossover_alpha * span)
            for i in (4, 5):  # discrete genes: uniform crossover
                child[i] = p1[i] if rng.uniform() < 0.5 else p2[i]

            if rng.uniform() < mutation_rate:
                bnd = list(BOUNDS.values())
                for i in range(4):
                    sigma = mutation_sigma_frac * (bnd[i][1] - bnd[i][0])
                    child[i] += rng.normal(0, sigma)
                if rng.uniform() < 0.3:
                    child[4] = rng.integers(0, len(GRADES))
                if rng.uniform() < 0.15:
                    child[5] = rng.integers(0, 2)

            new_pop.append(_clip_genome(child))
        pop = new_pop

    wall_time = time.time() - t0
    return dict(
        span_mm=span_mm, load_kNm=load_kNm, storey=storey,
        economy_metric=economy_metric, fitness=best_fitness, **best_meta,
        n_generations=n_generations, pop_size=pop_size, n_evaluations=fitness_evals,
        wall_time_s=wall_time,
    )


if __name__ == "__main__":
    r = ga_design(span_mm=8000, load_kNm=60, storey=20, seed=0)
    print(r)
