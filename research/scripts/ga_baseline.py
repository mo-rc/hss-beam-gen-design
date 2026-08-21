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


def ga_design_fixed_grade(span_mm, load_kNm, storey, grade, section_type, economy_metric="cost",
                            pop_size=60, n_generations=80, seed=0, tournament_k=3,
                            crossover_alpha=0.3, mutation_rate=0.25, mutation_sigma_frac=0.10,
                            elitism=2):
    """Same GA, but grade and section_type are FIXED (only the 4 continuous
    geometry genes evolve). Used exclusively by
    research/tests/validate_ground_truth.py to cross-check whether the
    brute-force grid search in generate_ec3_pretrain_dataset.py actually
    found the (grade, type)-conditional optimum it claims to -- i.e. is the
    thing everything else calls 'ground truth' actually trustworthy."""
    rng = np.random.default_rng(seed)
    env = _make_probe_env(economy_metric)
    grade_idx = int(np.argmin(np.abs(GRADES - grade)))
    type_idx = SECTION_TYPES.index(section_type)

    def random_genome_fixed():
        g = _random_genome(rng)
        g[4], g[5] = grade_idx, type_idx
        return g

    pop = [random_genome_fixed() for _ in range(pop_size)]
    best_genome, best_fitness, best_meta = None, -np.inf, None

    for gen in range(n_generations):
        scored = []
        for g in pop:
            fit, meta = _evaluate(g, env, span_mm, load_kNm, storey)
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
            for i in range(4):
                lo, hi = min(p1[i], p2[i]), max(p1[i], p2[i])
                span = hi - lo
                child[i] = rng.uniform(lo - crossover_alpha * span, hi + crossover_alpha * span)
            child[4], child[5] = grade_idx, type_idx  # fixed, never crossed/mutated
            if rng.uniform() < mutation_rate:
                bnd = list(BOUNDS.values())
                for i in range(4):
                    sigma = mutation_sigma_frac * (bnd[i][1] - bnd[i][0])
                    child[i] += rng.normal(0, sigma)
            new_pop.append(_clip_genome(child))
        pop = new_pop

    return dict(span_mm=span_mm, load_kNm=load_kNm, storey=storey,
                economy_metric=economy_metric, fitness=best_fitness, **best_meta)


def random_search_design(span_mm, load_kNm, storey, economy_metric="cost",
                           n_evaluations=4800, seed=0):
    """
    Random search using the SAME number of EC3 evaluations as the GA
    (pop_size * n_generations, by convention -- pass n_evaluations
    explicitly to match whatever GA budget you're comparing against).
    Directly answers the supervisor's explicit request: "Random search,
    using the same number of environment evaluations" -- tests whether
    GA's evolutionary structure (selection, crossover, mutation) earns its
    keep over pure i.i.d. sampling within the same bounds and budget.
    """
    rng = np.random.default_rng(seed)
    env = _make_probe_env(economy_metric)
    t0 = time.time()
    best_fitness, best_meta = -np.inf, None
    for _ in range(n_evaluations):
        g = _random_genome(rng)
        fit, meta = _evaluate(g, env, span_mm, load_kNm, storey)
        if fit > best_fitness:
            best_fitness, best_meta = fit, meta
    wall_time = time.time() - t0
    return dict(span_mm=span_mm, load_kNm=load_kNm, storey=storey,
                economy_metric=economy_metric, fitness=best_fitness, **best_meta,
                n_evaluations=n_evaluations, wall_time_s=wall_time)


def rule_based_design(span_mm, load_kNm, storey, economy_metric="cost", grade=355.0):
    """
    Deterministic, non-optimizing heuristic sizer, mimicking a first-pass
    manual design: start from a standard depth-to-span rule of thumb
    (h ~ span/24, a common serviceability-driven starting point for simply
    supported steel beams), default to the most commonly stocked grade
    (S355) unless told otherwise, then iteratively bump EVERY geometry
    variable up by a fixed increment (in a fixed priority order: depth,
    then flange thickness, then web thickness, then width) until EC3
    compliant, stopping at the first feasible design found -- exactly the
    "do the minimum needed, don't optimise further" behaviour a time-
    pressured engineer doing a first pass would exhibit. This directly
    answers the supervisor's request for a "rule-based EC3 design
    procedure or conventional engineering sizing" baseline -- the
    question this answers is not "can RL/GA find a good design" (they
    obviously can) but "how much is left on the table by NOT optimising
    at all, using only standard practice defaults".
    """
    env = _make_probe_env(economy_metric)
    span_m = span_mm / 1000.0
    h = float(np.clip(round(span_mm / 24.0 / 10) * 10, *BOUNDS["h"]))
    b = float(np.clip(round(h / 2.2 / 10) * 10, *BOUNDS["b"]))
    tf, tw = 10.0, 7.0
    section_type = "rolled"

    t0 = time.time()
    n_evals = 0
    max_iters = 200
    for _ in range(max_iters):
        env.h, env.b, env.tf, env.tw = h, b, tf, tw
        env.fy, env.section_type = grade, section_type
        env.span, env.load, env.storey = span_mm, load_kNm, storey
        util, mass, penalty, class_loss, chi_lt, dbg = env._ec3_analysis()
        n_evals += 1
        if util <= 1.0 and class_loss == 0 and penalty <= 1e-6:
            break
        # Fixed-priority bump order: depth first (cheapest capacity per kg
        # added, in general), then flange, then web, then width -- and if
        # depth has hit its bound, widen before anything else.
        if h < BOUNDS["h"][1]:
            h = min(h + 20.0, BOUNDS["h"][1])
        elif tf < BOUNDS["tf"][1]:
            tf = min(tf + 1.0, BOUNDS["tf"][1])
        elif tw < BOUNDS["tw"][1]:
            tw = min(tw + 1.0, BOUNDS["tw"][1])
        elif b < BOUNDS["b"][1]:
            b = min(b + 10.0, BOUNDS["b"][1])
        else:
            break  # exhausted all bounds, genuinely can't reach feasibility
    wall_time = time.time() - t0

    cost, co2, _ = env._calculate_cost_co2(mass)
    feasible = (util <= 1.0 + 1e-3) and (class_loss == 0) and (penalty <= 1e-6)
    return dict(span_mm=span_mm, load_kNm=load_kNm, storey=storey, grade=grade,
                section_type=section_type, economy_metric=economy_metric,
                h=h, b=b, tf=tf, tw=tw, util=util, mass=mass, cost=cost, co2=co2,
                feasible=feasible, n_evaluations=n_evals, wall_time_s=wall_time)


if __name__ == "__main__":
    r = ga_design(span_mm=8000, load_kNm=60, storey=20, seed=0)
    print("GA:", r)
    r2 = random_search_design(span_mm=8000, load_kNm=60, storey=20, n_evaluations=4800, seed=0)
    print("Random search:", r2)
    r3 = rule_based_design(span_mm=8000, load_kNm=60, storey=20)
    print("Rule-based:", r3)
