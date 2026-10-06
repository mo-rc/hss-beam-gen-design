"""
hssbeamgen/algo/de_search.py -- differential-evolution baseline (DE/rand/1/bin) with an EXACT
evaluation budget, on the same genome, physics and fitness as hssbeamgen/algo/ga_search.py.

Reuses ga_search's `_evaluate`, `_random_genome` and BOUNDS, so GA, random search and DE are
scored by identical code. Genome = [h, b, tf, tw, grade_idx, section_type_idx]; the two discrete
genes are kept continuous inside the search and rounded by `_evaluate` (standard DE practice).
A fixed budget `n_evaluations` is honoured exactly: the initial population consumes `pop_size`
evaluations, each generation another `pop_size`, and the last generation is truncated so the
total never exceeds the budget.
"""
import time

import numpy as np

from hssbeamgen.algo.ga_search import (BOUNDS, GRADES, _evaluate, _make_probe_env, _random_genome)

_LO = np.array([BOUNDS["h"][0], BOUNDS["b"][0], BOUNDS["tf"][0], BOUNDS["tw"][0], 0.0, 0.0])
_HI = np.array([BOUNDS["h"][1], BOUNDS["b"][1], BOUNDS["tf"][1], BOUNDS["tw"][1],
                float(len(GRADES) - 1), 1.0])


def de_design(span_mm, load_kN_per_m, storey, economy_metric="cost", n_evaluations=4800,
              pop_size=None, seed=0, F=0.6, CR=0.9):
    if pop_size is None:
        pop_size = int(np.clip(round(np.sqrt(n_evaluations)), 6, 60))
    if n_evaluations < pop_size:
        raise ValueError("n_evaluations must be >= pop_size")
    rng = np.random.default_rng(seed)
    env = _make_probe_env(economy_metric)
    t0 = time.time()
    n_eval = 0
    best_fit, best_meta = -np.inf, None

    pop = [_random_genome(rng) for _ in range(pop_size)]
    fit = np.empty(pop_size)
    for i, g in enumerate(pop):
        fit[i], meta = _evaluate(g, env, span_mm, load_kN_per_m, storey)
        n_eval += 1
        if fit[i] > best_fit:
            best_fit, best_meta = fit[i], meta
    pop = np.array(pop)

    while n_eval < n_evaluations:
        for i in range(pop_size):
            if n_eval >= n_evaluations:
                break
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
            mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), _LO, _HI)
            cross = rng.uniform(size=6) < CR
            cross[rng.integers(0, 6)] = True
            trial = np.where(cross, mutant, pop[i])
            f, meta = _evaluate(trial, env, span_mm, load_kN_per_m, storey)
            n_eval += 1
            if f >= fit[i]:
                pop[i], fit[i] = trial, f
            if f > best_fit:
                best_fit, best_meta = f, meta

    return dict(span_mm=span_mm, load_kN_per_m=load_kN_per_m, storey=storey,
                economy_metric=economy_metric, fitness=best_fit, **best_meta,
                pop_size=pop_size, n_evaluations=n_eval, wall_time_s=time.time() - t0)
