"""
research/scripts/expA_strong_baselines.py
================================================================
Exp A -- strong optimisation baselines: quality vs EC3-evaluation budget.

STANDALONE diagnostic / benchmark. It imports the environment's physics
(HSSBeamEnv._ec3_analysis / _calculate_cost_co2 / _constraint_violations /
_economy) UNCHANGED, reads the existing ground-truth CSVs read-only, and
writes only to a NEW directory (default research/results/expA/). It does not
touch any existing result, script or environment file.

QUESTION
--------
For the SAME 142 contexts, the SAME EC3 physics, objective, feasibility
definition and ground-truth reference: what optimality gap do GA, DE, CMA-ES
and multistart local search reach as a function of the number of EC3
evaluations (25 ... 4800), and how much does the design-space
parameterisation (raw vs reparameterised) matter? This is the target curve
any RL method must beat under an equal-evaluation-budget comparison.

WHAT IS COUNTED AS AN "EC3 EVALUATION"
--------------------------------------
Exactly one call of env._ec3_analysis(). The meter wraps that method on the
environment instance the optimisers use, so NOTHING can evaluate a design
without being counted: optimiser proposals, finite-difference steps inside
SLSQP, and every sizing step of the reparameterised decoder are all counted.
  * Each (context, method, space, budget, seed) run has a HARD cap B; the run
    is stopped the moment B evaluations have been spent (mid-generation if
    necessary). A run never exceeds B (asserted).
  * The reported design is the best FEASIBLE design among the designs that were
    actually evaluated. No uncounted final evaluation is ever made.
  * n_direct = evaluations of optimiser proposals, n_sizing = evaluations
    spent by the decoder's boundary-sizing (0 in the raw space).
  * Optimiser hyper-parameters that depend on the budget (population size,
    number of starts) are set FROM B; results for B=25 are NOT read off a
    B=4800 run.
  * Identical inputs inside one local solve are memoised (SLSQP asks for the
    objective and the constraint at the same x); the physics call happens
    once and is counted once.

DESIGN SPACES (same underlying design space, two coordinate systems)
-------------------------------------------------------------------
  raw      : (h, b, tf, tw, grade, type) in the environment's box.
  reparam  : (h, b/h, lambda_f, lambda_w, grade, type). tf, tw are DERIVED from
             the EC3 Table 5.2 slenderness ratios (class <= 3 by construction
             when the thickness bounds are not active), then the whole section
             is uniformly scaled to the utilisation boundary by a
             safeguarded log-log secant search (<= size_max EC3 evaluations,
             stops once feasible and util >= 1 - size_tol). All of these
             evaluations are counted. The reparam box is a SEARCH box only:
             it contains every ground-truth optimum (checked at start-up) and
             excludes no feasible design that matters; EC3 physics decides
             feasibility, never the decoder.
  Categorical genes: grade in {355,460,500,550,620,690}, type in
  {rolled, welded}; continuous relaxations use u -> floor(u * n_levels).

METHODS (all normalised to u in [0,1]^d, all restart until the budget is spent)
--------------------------------------------------------------------------------
  ga        real-coded GA, tournament k=3, BLX-0.3 / uniform crossover,
            gaussian (0.1) / reset mutation, elitism 2, Deb feasibility rules;
            pop = clip(round(pop_mult*sqrt(B)), 6, 200); pop_mult tuned (default 1).
  de        scipy differential_evolution (best1bin, dither F in (0.5,1),
            CR 0.9, LHS init, immediate updating, no polish), population
            6*round(NP/6), NP as for GA; restarted with a new seed if it
            converges before the budget is spent.
  cmaes     cma.CMAEvolutionStrategy, sigma0 = 0.3, random initial mean,
            popsize = clip(round(pop_mult*sqrt(B)), 6, 96); IPOP-style restart
            (popsize x2, new random mean) on internal stop.
  ms_nm     multistart bounded Nelder-Mead on the 4 continuous coordinates,
            categorical combination cycled through all 12 (grade,type) pairs;
            #starts target = clip(B // start_evals, 1, 64), per-start cap = ceil(B/#starts)
            evaluations, more starts if a start converges early.
  ms_slsqp  as ms_nm but SLSQP with finite-difference gradients (raw space:
            objective = economy, constraint = 1 - util - class/geometry
            violation >= 0; reparam space: unconstrained, sizing enforces util<=1).
  Penalised scalar objective for de / cmaes / ms_nm: economy/1e3 if feasible,
  else 100 + 100 * (g1 + g2 + g3). GA uses Deb's rules directly.

  HYPER-PARAMETERS: one scalar per (method, space) -- pop_mult for ga/de/cmaes,
  start_evals for the multistart methods -- was selected with `--tune` on a
  12-context subset (context_seed 777) using seeds disjoint from the final run
  (base_seed 1000), scoring the mean gap over budgets {100,400,1600} with an
  infeasible run counted as gap=3.0. The subset overlaps the 142 evaluation
  contexts (the GT exists only on this grid), which if anything favours the
  baselines. Nothing is tuned per context or per budget. The tuning log is
  expA_tuning_log.csv; the frozen values are DEFAULT_HP and are written to
  the meta JSON of every run. `--tune` regenerates them.

METRIC
------
gap = best_feasible_economy / GT_optimum_economy - 1, GT optimum taken per
(span, load) as min over (grade, type) rows of the objective-specific
ground-truth file -- identical to research/scripts/evaluate.py. Negative gaps
(design better than the GA-generated ground truth) are kept and reported.
Runs with no feasible design are counted in `feasibility_rate` and excluded
from gap statistics; `frac_gap_le_*` uses ALL runs (failures count as misses).

USAGE (see the bottom of the file / final report for the exact commands)
    python research/scripts/expA_strong_baselines.py --tag full --n_jobs 4
    python research/scripts/expA_strong_baselines.py --tag full --resume --n_jobs 4
    python research/scripts/expA_strong_baselines.py --tag full --aggregate_only
================================================================
"""
import os
import sys
import json
import math
import time
import argparse
import platform
import subprocess
from datetime import datetime, timezone
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from research.envs.hss_env import HSSBeamEnv

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
LIM = dict(h=(250.0, 750.0), b=(120.0, 300.0), tf=(8.0, 35.0), tw=(6.0, 25.0))
GRADES = [355.0, 460.0, 500.0, 550.0, 620.0, 690.0]
TYPES = ["rolled", "welded"]
REP_LIM = dict(bh=(0.10, 0.80), lf=(3.0, 14.0), lw=(30.0, 124.0))
DIM = 6
METHODS_ALL = ["ga", "de", "cmaes", "ms_nm", "ms_slsqp"]
SPACES_ALL = ["raw", "reparam"]
# Per-(method, space) hyper-parameters. Selected with `--tune` on a held-out 12-context
# subset with different seeds than the final run (see expA_*_tuning.csv). Missing key -> 1.0 / 40.
DEFAULT_HP = {
    "ga|raw": {"pop_mult": 1.0},        "ga|reparam": {"pop_mult": 0.5},
    "de|raw": {"pop_mult": 1.0},        "de|reparam": {"pop_mult": 1.0},
    "cmaes|raw": {"pop_mult": 0.5},     "cmaes|reparam": {"pop_mult": 0.5},
    "ms_nm|raw": {"start_evals": 100},  "ms_nm|reparam": {"start_evals": 25},
    "ms_slsqp|raw": {"start_evals": 25}, "ms_slsqp|reparam": {"start_evals": 25},
}
F_SCALE = 1.0e3          # economy normalisation inside penalised objectives
F_INFEAS = 100.0         # base value of the penalised objective when infeasible
FEAS_TOL = 1e-3          # identical to HSSBeamEnv.step()'s `feasible`


class BudgetExhausted(Exception):
    """Raised by the meter when the hard EC3-evaluation cap B is reached."""


class StartBudget(Exception):
    """Raised inside a multistart local search when one start's cap is reached."""


def _cat(u4, u5):
    gi = min(int(u4 * len(GRADES)), len(GRADES) - 1)
    ti = min(int(u5 * 2), 1)
    return GRADES[gi], TYPES[ti]


# ----------------------------------------------------------------------------
# EC3 evaluation meter: the single place where physics is called and counted
# ----------------------------------------------------------------------------
class Ec3Meter:
    def __init__(self, env, metric):
        self.env, self.metric = env, metric
        self._orig = env._ec3_analysis
        env._ec3_analysis = self._counted          # every physics call goes through here
        self.reset(None, None, math.inf)

    def reset(self, span_mm, load, budget):
        self.span, self.load, self.budget = span_mm, load, budget
        self.n = self.n_direct = self.n_sizing = self.n_feas = 0
        self.phase = "direct"
        self.best_econ, self.best_design, self.n_at_best = None, None, None

    def _counted(self):
        if self.n >= self.budget:
            raise BudgetExhausted()
        self.n += 1
        if self.phase == "direct":
            self.n_direct += 1
        else:
            self.n_sizing += 1
        return self._orig()

    def evaluate(self, h, b, tf, tw, fy, st):
        """One EC3 evaluation of a concrete design (counted)."""
        env = self.env
        env.use_storey_load_scaling = False
        env.span, env.load = float(self.span), float(self.load)
        env.h, env.b, env.tf, env.tw = float(h), float(b), float(tf), float(tw)
        env.fy, env.section_type = float(fy), st
        util, mass, pen, closs, _chi, dbg = env._ec3_analysis()      # <- counted
        cost, co2, _ = env._calculate_cost_co2(mass)
        v = env._constraint_violations(util, closs, pen)
        feas = all(x <= FEAS_TOL for x in v.values())
        # raw objective value (what the GT files store); env._economy() is the same
        # quantity divided by a constant normaliser, so ordering/feasibility are identical.
        econ = float({"mass": mass, "cost": cost, "co2": co2}[self.metric])
        res = dict(feas=feas, econ=econ, util=float(util), viol=float(sum(v.values())),
                   g23=float(v["g2_class"] + v["g3_geom"]))
        if feas:
            self.n_feas += 1
            if self.best_econ is None or econ < self.best_econ:
                self.best_econ, self.n_at_best = econ, self.n
                self.best_design = (float(h), float(b), float(tf), float(tw), float(fy), st)
        return res


def pen_obj(res):
    """Penalised scalar objective (minimise)."""
    return res["econ"] / F_SCALE if res["feas"] else F_INFEAS + F_INFEAS * res["viol"]


# ----------------------------------------------------------------------------
# Design spaces
# ----------------------------------------------------------------------------
class RawSpace:
    name = "raw"

    def propose(self, u, m):
        h = LIM["h"][0] + u[0] * (LIM["h"][1] - LIM["h"][0])
        b = LIM["b"][0] + u[1] * (LIM["b"][1] - LIM["b"][0])
        tf = LIM["tf"][0] + u[2] * (LIM["tf"][1] - LIM["tf"][0])
        tw = LIM["tw"][0] + u[3] * (LIM["tw"][1] - LIM["tw"][0])
        fy, st = _cat(u[4], u[5])
        m.phase = "direct"
        return m.evaluate(h, b, tf, tw, fy, st)


class ReparamSpace:
    name = "reparam"

    def __init__(self, size_tol=0.003, size_max=10):
        self.size_tol, self.size_max = size_tol, size_max

    @staticmethod
    def decode(u):
        h = LIM["h"][0] + u[0] * (LIM["h"][1] - LIM["h"][0])
        bh = REP_LIM["bh"][0] + u[1] * (REP_LIM["bh"][1] - REP_LIM["bh"][0])
        lf = REP_LIM["lf"][0] + u[2] * (REP_LIM["lf"][1] - REP_LIM["lf"][0])
        lw = REP_LIM["lw"][0] + u[3] * (REP_LIM["lw"][1] - REP_LIM["lw"][0])
        fy, st = _cat(u[4], u[5])
        eps = math.sqrt(235.0 / fy)
        b = float(np.clip(bh * h, *LIM["b"]))
        tf, tw = 15.0, 8.0
        for _ in range(12):                                   # fixed point (rolled root radius)
            r = 0.1 * tf if st == "rolled" else 0.0
            tf_new = float(np.clip(((b - tw) / 2.0 - r) / (lf * eps), *LIM["tf"]))
            r2 = 0.1 * tf_new if st == "rolled" else 0.0
            tw_new = float(np.clip(((h - 2 * tf_new) - 2 * r2) / (lw * eps), *LIM["tw"]))
            done = abs(tf_new - tf) < 1e-6 and abs(tw_new - tw) < 1e-6
            tf, tw = tf_new, tw_new
            if done:
                break
        return (h, b, tf, tw), fy, st

    def propose(self, u, m):
        (h, b, tf, tw), fy, st = self.decode(u)

        def ev(s, phase):
            m.phase = phase
            g = (np.clip(h * s, *LIM["h"]), np.clip(b * s, *LIM["b"]),
                 np.clip(tf * s, *LIM["tf"]), np.clip(tw * s, *LIM["tw"]))
            return m.evaluate(*g, fy, st)

        tgt = math.log(1.0 - 0.5 * self.size_tol)
        res = last = ev(1.0, "direct")
        best_feas = res if res["feas"] else None
        lo = hi = None                                       # (s, util): lo infeasible, hi feasible
        if res["feas"]:
            hi = (1.0, res["util"])
        else:
            lo = (1.0, res["util"])
        s, n = 1.0, 1
        while n < self.size_max:
            if last["feas"] and last["util"] >= 1.0 - self.size_tol:
                break
            if lo is not None and hi is not None:
                x_lo, y_lo = math.log(lo[0]), math.log(max(lo[1], 1e-9))
                x_hi, y_hi = math.log(hi[0]), math.log(max(hi[1], 1e-9))
                if abs(y_lo - y_hi) < 1e-12:
                    x = 0.5 * (x_lo + x_hi)
                else:
                    x = x_hi + (tgt - y_hi) * (x_lo - x_hi) / (y_lo - y_hi)
                a, c = min(x_lo, x_hi), max(x_lo, x_hi)
                x = float(np.clip(x, a + 0.1 * (c - a), c - 0.1 * (c - a)))
                s_new = math.exp(x)
            else:                                            # power-law guess util ~ s^-3
                s_new = s * (max(last["util"], 1e-3) / (1.0 - 0.5 * self.size_tol)) ** (1.0 / 3.0)
                s_new = float(np.clip(s_new, 0.2, 4.0))
            s = s_new
            last = ev(s, "sizing")
            n += 1
            if last["feas"]:
                if best_feas is None or last["econ"] < best_feas["econ"]:
                    best_feas = last
                if hi is None or s < hi[0]:
                    hi = (s, last["util"])
            else:
                if lo is None or s > lo[0]:
                    lo = (s, last["util"])
        return best_feas if best_feas is not None else last


# ----------------------------------------------------------------------------
# Optimisers (all minimise the penalised objective, or use Deb rules)
# ----------------------------------------------------------------------------
def _pop_size(B, mult=1.0, cap=200):
    return int(np.clip(round(mult * math.sqrt(B)), 6, cap))


def run_ga(space, m, B, rng, cfg):
    P = _pop_size(B, cfg.get("pop_mult", 1.0))
    tour_k, alpha, mut_rate, mut_sig, cat_rate, elite = 3, 0.3, 0.25, 0.10, 0.20, 2
    key = lambda r: (0, r["econ"]) if r["feas"] else (1, r["viol"])
    pop = [rng.random(DIM) for _ in range(P)]
    fit = [key(space.propose(u, m)) for u in pop]
    while True:
        order = sorted(range(P), key=lambda i: fit[i])
        new_pop = [pop[i].copy() for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        while len(new_pop) < P:
            def pick():
                cand = rng.integers(0, P, size=tour_k)
                return pop[min(cand, key=lambda i: fit[i])]
            p1, p2 = pick(), pick()
            c = np.empty(DIM)
            for j in range(4):
                lo_, hi_ = min(p1[j], p2[j]), max(p1[j], p2[j])
                d = hi_ - lo_
                c[j] = rng.uniform(lo_ - alpha * d, hi_ + alpha * d)
            for j in (4, 5):
                c[j] = p1[j] if rng.random() < 0.5 else p2[j]
            for j in range(4):
                if rng.random() < mut_rate:
                    c[j] += rng.normal(0.0, mut_sig)
            for j in (4, 5):
                if rng.random() < cat_rate:
                    c[j] = rng.random()
            c = np.clip(c, 0.0, 1.0 - 1e-9)
            new_pop.append(c)
            new_fit.append(key(space.propose(c, m)))
        pop, fit = new_pop, new_fit


def run_de(space, m, B, rng, cfg):
    NP = _pop_size(B, cfg.get("pop_mult", 1.0))
    popsize = max(1, int(round(NP / DIM)))
    f = lambda u: pen_obj(space.propose(np.clip(u, 0.0, 1.0 - 1e-9), m))
    while True:                                              # restart on early convergence
        differential_evolution(f, [(0.0, 1.0)] * DIM, strategy="best1bin", popsize=popsize,
                               maxiter=10 ** 9, tol=0.0, atol=0.0, mutation=(0.5, 1.0),
                               recombination=0.9, polish=False, init="latinhypercube",
                               seed=int(rng.integers(2 ** 31 - 1)), updating="immediate",
                               disp=False, workers=1)


def run_cmaes(space, m, B, rng, cfg):
    import cma
    pop = _pop_size(B, cfg.get("pop_mult", 1.0), cap=96)
    f = lambda u: pen_obj(space.propose(np.clip(u, 0.0, 1.0 - 1e-9), m))
    while True:                                              # IPOP-style restarts
        es = cma.CMAEvolutionStrategy(rng.random(DIM), 0.3,
                                      dict(bounds=[0.0, 1.0], popsize=pop, verbose=-9,
                                           seed=int(rng.integers(1, 2 ** 31 - 1))))
        while not es.stop():
            X = es.ask()
            es.tell(X, [f(np.asarray(x)) for x in X])
        pop = min(2 * pop, 128)


def _multistart(space, m, B, rng, cfg, local):
    n_starts = int(np.clip(B // int(cfg.get("start_evals", 40)), 1, 64))
    per_start = int(math.ceil(B / n_starts))
    combos = [(g, t) for g in range(len(GRADES)) for t in range(2)]
    rng.shuffle(combos)
    k = 0
    while True:
        gi, ti = combos[k % len(combos)]
        k += 1
        cat = np.array([(gi + 0.5) / len(GRADES), (ti + 0.5) / 2.0])
        x0 = rng.random(4)
        limit = m.n + per_start
        cache = {}

        def full(uc):
            return np.concatenate([np.clip(uc, 0.0, 1.0), cat])

        def ev(uc):
            key = np.asarray(uc, dtype=float).tobytes()
            if key in cache:
                return cache[key]
            if m.n >= limit:
                raise StartBudget()
            r = space.propose(full(uc), m)
            if len(cache) > 512:
                cache.clear()
            cache[key] = r
            return r

        try:
            if local == "nm":
                minimize(lambda uc: pen_obj(ev(uc)), x0, method="Nelder-Mead",
                         bounds=[(0.0, 1.0)] * 4,
                         options=dict(maxfev=10 ** 7, xatol=1e-4, fatol=1e-7, adaptive=True))
            else:
                if space.name == "raw":
                    obj = lambda uc: ev(uc)["econ"] / F_SCALE
                    con = [dict(type="ineq", fun=lambda uc: 1.0 - ev(uc)["util"] - ev(uc)["g23"])]
                    eps = 1e-3
                else:
                    obj = lambda uc: pen_obj(ev(uc))
                    con = []
                    eps = 2e-2
                minimize(obj, x0, method="SLSQP", bounds=[(0.0, 1.0)] * 4, constraints=con,
                         options=dict(maxiter=200, ftol=1e-8, eps=eps))
        except StartBudget:
            pass


def run_ms_nm(space, m, B, rng, cfg):
    _multistart(space, m, B, rng, cfg, "nm")


def run_ms_slsqp(space, m, B, rng, cfg):
    _multistart(space, m, B, rng, cfg, "slsqp")


RUNNERS = dict(ga=run_ga, de=run_de, cmaes=run_cmaes, ms_nm=run_ms_nm, ms_slsqp=run_ms_slsqp)


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------
_W = {}


def _init_worker(metric, size_tol, size_max):
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=metric,
                     enforce_rolled_manufacturability=True)
    _W["meter"] = Ec3Meter(env, metric)
    _W["spaces"] = dict(raw=RawSpace(), reparam=ReparamSpace(size_tol, size_max))


def _task(a):
    ctx_i, span_m, load, gt, method, space, budgets, n_seeds, base_seed, hp = a
    m, sp = _W["meter"], _W["spaces"][space]
    mid, sid = METHODS_ALL.index(method), SPACES_ALL.index(space)
    rows = []
    for bi, B in enumerate(budgets):
        for s in range(n_seeds):
            m.reset(span_m * 1000.0, load, B)
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, ctx_i, mid, sid, B, s]))
            t0 = time.perf_counter()
            try:
                RUNNERS[method](sp, m, B, rng, hp)
            except BudgetExhausted:
                pass
            dt = time.perf_counter() - t0
            assert m.n <= B and m.n == m.n_direct + m.n_sizing, "evaluation accounting violated"
            feas = m.best_econ is not None
            row = dict(context_idx=ctx_i, span_m=span_m, load_kNm=load, gt_economy=gt, method=method,
                       space=space, budget=B, seed=s, feasible=int(feas),
                       best_economy=m.best_econ if feas else np.nan,
                       gap=(m.best_econ / gt - 1.0) if feas else np.nan,
                       n_evals=m.n, n_direct=m.n_direct, n_sizing=m.n_sizing, n_feasible_evals=m.n_feas,
                       n_at_best=m.n_at_best if feas else -1, wall_s=dt, hp=json.dumps(hp, sort_keys=True))
            if feas:
                row.update(zip(["h", "b", "tf", "tw", "fy", "section_type"], m.best_design))
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Ground truth, sanity checks, aggregation
# ----------------------------------------------------------------------------
def load_contexts(gt_dir, metric, n_contexts, ctx_seed):
    gt = pd.read_csv(os.path.join(gt_dir, f"ec3_optimal_designs_{metric}.csv"))
    opt = gt.loc[gt.groupby(["span_m", "load_kNm"])[metric].idxmin()].reset_index(drop=True)
    if n_contexts is not None and n_contexts < len(opt):
        opt = opt.sample(n=n_contexts, random_state=ctx_seed).sort_index().reset_index(drop=True)
    return opt


def gt_sanity_check(opt, metric, size_tol, size_max):
    """(1) GT designs re-evaluate to the stored objective and are feasible under
    the same physics (catches GT/physics pairing bugs). (2) every GT optimum
    lies inside the reparam search box. Uses its own, unbudgeted meter."""
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=metric,
                     enforce_rolled_manufacturability=True)
    m = Ec3Meter(env, metric)
    worst, infeas, outside = 0.0, 0, 0
    for r in opt.itertuples():
        m.reset(r.span_m * 1000.0, r.load_kNm, math.inf)
        res = m.evaluate(r.h, r.b, r.tf, r.tw, r.grade, r.section_type)
        worst = max(worst, abs(res["econ"] / getattr(r, metric) - 1.0))
        infeas += int(not res["feas"])
        eps = math.sqrt(235.0 / r.grade)
        rr = 0.1 * r.tf if r.section_type == "rolled" else 0.0
        lf = ((r.b - r.tw) / 2.0 - rr) / (r.tf * eps)
        lw = ((r.h - 2 * r.tf) - 2 * rr) / (r.tw * eps)
        bh = r.b / r.h
        outside += int(not (REP_LIM["lf"][0] <= lf <= REP_LIM["lf"][1] and REP_LIM["lw"][0] <= lw <= REP_LIM["lw"][1]
                            and REP_LIM["bh"][0] <= bh <= REP_LIM["bh"][1]))
    return dict(n_contexts=len(opt), max_rel_dev_gt_reeval=worst, gt_designs_infeasible=infeas,
                gt_optima_outside_reparam_box=outside)


def summarize(runs):
    runs = runs.drop_duplicates(["context_idx", "method", "space", "budget", "seed"])
    recs = []
    for (method, space, B), d in runs.groupby(["method", "space", "budget"]):
        f = d[d.feasible == 1]
        g = f.gap.values
        seed_means = f.groupby("seed").gap.mean()
        ctx_sd = f.groupby("context_idx").gap.std(ddof=1)
        recs.append(dict(
            method=method, space=space, budget=int(B),
            n_runs=len(d), n_contexts=d.context_idx.nunique(), n_seeds=d.seed.nunique(),
            feasibility_rate=float(d.feasible.mean()),
            gap_mean=float(np.mean(g)) if len(g) else np.nan,
            gap_median=float(np.median(g)) if len(g) else np.nan,
            gap_p90=float(np.percentile(g, 90)) if len(g) else np.nan,
            gap_p95=float(np.percentile(g, 95)) if len(g) else np.nan,
            gap_max=float(np.max(g)) if len(g) else np.nan,
            gap_std=float(np.std(g, ddof=1)) if len(g) > 1 else np.nan,
            seedmean_mean=float(seed_means.mean()) if len(seed_means) else np.nan,
            seedmean_sd=float(seed_means.std(ddof=1)) if len(seed_means) > 1 else np.nan,
            seedmean_min=float(seed_means.min()) if len(seed_means) else np.nan,
            seedmean_max=float(seed_means.max()) if len(seed_means) else np.nan,
            within_ctx_sd_mean=float(ctx_sd.mean()) if len(ctx_sd) else np.nan,
            frac_gap_le_2pct=float(((d.gap <= 0.02) & (d.feasible == 1)).mean()),
            frac_gap_le_5pct=float(((d.gap <= 0.05) & (d.feasible == 1)).mean()),
            mean_evals=float(d.n_evals.mean()), max_evals=int(d.n_evals.max()),
            mean_sizing_frac=float((d.n_sizing / d.n_evals).mean()),
            mean_wall_s=float(d.wall_s.mean())))
    order = {m: i for i, m in enumerate(METHODS_ALL)}
    out = pd.DataFrame(recs)
    out["_o"] = out.method.map(order)
    return out.sort_values(["space", "_o", "budget"]).drop(columns="_o").reset_index(drop=True)


def hp_for(hp_table, method, space):
    return dict(hp_table.get(f"{method}|{space}", {}))


TUNE_GRID = dict(
    ga=[dict(pop_mult=x) for x in (0.5, 1.0, 2.0)],
    de=[dict(pop_mult=x) for x in (0.5, 1.0, 2.0)],
    cmaes=[dict(pop_mult=x) for x in (0.25, 0.5, 1.0, 2.0)],
    ms_nm=[dict(start_evals=x) for x in (25, 50, 100, 200)],
    ms_slsqp=[dict(start_evals=x) for x in (25, 50, 100, 200)],
)


def run_tune(a, methods, spaces, pool_factory):
    """Select each (method, space) hyper-parameter on a held-out subset of contexts
    with seeds DISJOINT from the final run (base_seed = a.tune_base_seed).
    Score = mean over tuning runs of gap (infeasible run counts as gap = 3.0)."""
    opt = load_contexts(a.gt_dir, a.metric, a.tune_contexts, a.tune_context_seed)
    tb = [int(x) for x in a.tune_budgets.split(",")]
    tasks, tags = [], []
    for mth in methods:
        for sp in spaces:
            for hp in TUNE_GRID[mth]:
                for i, r in enumerate(opt.itertuples()):
                    tasks.append((i, r.span_m, r.load_kNm, float(getattr(r, a.metric)), mth, sp, tb,
                                  a.tune_seeds, a.tune_base_seed, hp))
    rows = []
    for out in pool_factory(tasks):
        rows.extend(out)
    d = pd.DataFrame(rows)
    d["score_gap"] = np.where(d.feasible == 1, d.gap, 3.0)
    tun = (d.groupby(["method", "space", "hp"]).agg(score=("score_gap", "mean"), median=("score_gap", "median"),
                                                    feas=("feasible", "mean"), runs=("gap", "size")).reset_index())
    best = tun.loc[tun.groupby(["method", "space"]).score.idxmin()]
    hp_json = {f"{r.method}|{r.space}": json.loads(r.hp) for r in best.itertuples()}
    return tun, hp_json


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def print_table(summ):
    cols = ["method", "space", "budget", "feasibility_rate", "gap_mean", "gap_median", "gap_p90", "seedmean_sd"]
    t = summ[cols].copy()
    for c in ("gap_mean", "gap_median", "gap_p90", "seedmean_sd"):
        t[c] = (100 * t[c]).round(2)
    print(t.to_string(index=False))
    print("(gap columns in %; seedmean_sd = SD across seeds of the per-seed mean gap)")


def main():
    ap = argparse.ArgumentParser(description="Exp A: strong optimisation baselines vs EC3-evaluation budget")
    ap.add_argument("--metric", default="cost", choices=["mass", "cost", "co2"])
    ap.add_argument("--gt_dir", default="research/pretrain_data_corrected")
    ap.add_argument("--out_dir", default="research/results/expA")
    ap.add_argument("--tag", default="full", help="suffix of all output files (use a different tag for smoke tests)")
    ap.add_argument("--methods", default=",".join(METHODS_ALL))
    ap.add_argument("--spaces", default=",".join(SPACES_ALL))
    ap.add_argument("--budgets", default="25,50,100,400,1600,4800")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_contexts", type=int, default=None, help="default: all 142 contexts")
    ap.add_argument("--context_seed", type=int, default=0)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--size_tol", type=float, default=0.003)
    ap.add_argument("--size_max", type=int, default=10)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--hp_json", default=None, help="JSON {'method|space': {'pop_mult'|'start_evals': v}} overriding DEFAULT_HP")
    ap.add_argument("--tune", action="store_true", help="select hyper-parameters on held-out contexts/seeds, write expA_<tag>_hp.json, exit")
    ap.add_argument("--tune_contexts", type=int, default=12)
    ap.add_argument("--tune_context_seed", type=int, default=777)
    ap.add_argument("--tune_seeds", type=int, default=2)
    ap.add_argument("--tune_base_seed", type=int, default=1000)
    ap.add_argument("--tune_budgets", default="100,400,1600")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--skip_gt_check", action="store_true")
    a = ap.parse_args()

    methods, spaces = a.methods.split(","), a.spaces.split(",")
    budgets = [int(x) for x in a.budgets.split(",")]
    assert all(x in METHODS_ALL for x in methods) and all(x in SPACES_ALL for x in spaces)
    os.makedirs(a.out_dir, exist_ok=True)
    p = lambda s: os.path.join(a.out_dir, f"expA_{a.tag}_{s}")
    runs_csv = p("runs.csv")

    hp_table = dict(DEFAULT_HP)
    if a.hp_json:
        with open(a.hp_json) as fh:
            hp_table.update(json.load(fh))

    def make_iter(tasks):
        init = (a.metric, a.size_tol, a.size_max)
        if a.n_jobs > 1:
            pool = Pool(a.n_jobs, initializer=_init_worker, initargs=init)
            return pool.imap_unordered(_task, tasks)
        _init_worker(*init)
        return map(_task, tasks)

    if a.tune:
        t0 = time.time()
        tun, hpj = run_tune(a, methods, spaces, make_iter)
        tun.to_csv(p("tuning.csv"), index=False)
        with open(p("hp.json"), "w") as fh:
            json.dump(hpj, fh, indent=2)
        print(tun.to_string(index=False))
        print("\nselected:", json.dumps(hpj), f"\n({(time.time() - t0) / 60:.1f} min)  wrote {p('tuning.csv')}, {p('hp.json')}")
        return

    if not a.aggregate_only:
        opt = load_contexts(a.gt_dir, a.metric, a.n_contexts, a.context_seed)
        if os.path.exists(runs_csv) and not a.resume:
            sys.exit(f"{runs_csv} exists. Use --resume to continue it, or a new --tag. "
                     f"(Refusing to overwrite existing results.)")
        chk = None
        if not a.skip_gt_check:
            chk = gt_sanity_check(opt, a.metric, a.size_tol, a.size_max)
            print("GT sanity check:", chk)
            if chk["max_rel_dev_gt_reeval"] > 1e-3 or chk["gt_designs_infeasible"] > 0:
                sys.exit("GT designs do not re-evaluate to the stored objective / feasibility under this "
                         "physics. Check --gt_dir / --metric pairing. Aborting.")
            if chk["gt_optima_outside_reparam_box"] > 0:
                print("WARNING: some GT optima lie outside the reparam search box (see REP_LIM).")
        done = set()
        if os.path.exists(runs_csv):
            prev = pd.read_csv(runs_csv)
            prev = prev.drop_duplicates(["context_idx", "method", "space", "budget", "seed"])
            cnt = prev.groupby(["context_idx", "method", "space"]).size()
            need = len(budgets) * a.n_seeds
            done = {k for k, v in cnt.items() if v >= need}
            keep = prev.set_index(["context_idx", "method", "space"]).index.isin(list(done))
            if not keep.all() or len(prev) != sum(1 for _ in open(runs_csv)) - 1:
                print(f"resume: dropping {int((~keep).sum())} rows of incomplete tasks / duplicates")
                prev[keep].to_csv(runs_csv, index=False)
        tasks = [(i, r.span_m, r.load_kNm, float(getattr(r, a.metric)), mth, sp, budgets, a.n_seeds, a.base_seed,
                  hp_for(hp_table, mth, sp))
                 for i, r in enumerate(opt.itertuples()) for mth in methods for sp in spaces
                 if (i, mth, sp) not in done]
        meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), args=vars(a), git_commit=_git_commit(),
                    python=platform.python_version(), platform=platform.platform(),
                    versions={k: __import__(k).__version__ for k in ("numpy", "scipy", "pandas", "cma")},
                    gt_check=chk, n_contexts=len(opt), hp_table=hp_table,
                    hyperparameters=dict(
                        ga=dict(pop="clip(round(pop_mult*sqrt(B)),6,200)", tournament=3, blx_alpha=0.3, mut_rate=0.25,
                                mut_sigma=0.10, cat_reset=0.20, elitism=2, constraint_handling="Deb rules"),
                        de=dict(strategy="best1bin", pop="6*round(NP/6), NP=clip(round(pop_mult*sqrt(B)),6,200)",
                                F="(0.5,1.0)", CR=0.9, init="latinhypercube", updating="immediate", polish=False),
                        cmaes=dict(sigma0=0.3, popsize="clip(round(pop_mult*sqrt(B)),6,96)", restarts="IPOP x2 (cap 128)"),
                        ms=dict(n_starts_target="clip(B//start_evals,1,64)", per_start_cap="ceil(B/n_starts)",
                                nm="bounded adaptive Nelder-Mead", slsqp_fd_eps=dict(raw=1e-3, reparam=2e-2)),
                        reparam=dict(box=REP_LIM, size_tol=a.size_tol, size_max=a.size_max,
                                     sizing="safeguarded log-log secant, all evals counted"),
                        objective=dict(scale=F_SCALE, infeasible_base=F_INFEAS, feasibility_tol=FEAS_TOL)))
        with open(p("meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        print(f"{len(opt)} contexts | {len(tasks)} tasks ({len(done)} already done) | "
              f"budgets={budgets} seeds={a.n_seeds} methods={methods} spaces={spaces} jobs={a.n_jobs}")
        t0, header = time.time(), not os.path.exists(runs_csv)
        it = make_iter(tasks)
        for k, rows in enumerate(it, 1):
            pd.DataFrame(rows).to_csv(runs_csv, mode="a", header=header, index=False)
            header = False
            if k % max(1, len(tasks) // 50) == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"  {k}/{len(tasks)} tasks | {el / 60:.1f} min elapsed | ETA {(el / k) * (len(tasks) - k) / 60:.1f} min",
                      flush=True)

    runs = pd.read_csv(runs_csv)
    summ = summarize(runs)
    summ.to_csv(p("summary.csv"), index=False)
    with open(p("summary.json"), "w") as fh:
        json.dump(json.loads(summ.to_json(orient="records")), fh, indent=2)
    print_table(summ)
    print(f"\nwrote: {runs_csv}\n       {p('summary.csv')}\n       {p('summary.json')}\n       {p('meta.json')}")


if __name__ == "__main__":
    main()
