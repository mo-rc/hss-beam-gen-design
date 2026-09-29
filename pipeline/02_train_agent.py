"""Step 2 - train one RL run (PPO / SAC / TD3 / DDPG). One entrypoint, one CLI, for every
algorithm -- there is no per-algorithm script to keep in sync.

Merges the old train.py, train_baseline_offpolicy.py, resume_training.py and
manual_resume_from_checkpoint.py into one entrypoint. run_multiseed.py's seed loop is now a
shell loop (see pipeline plan sections 2a/2b/2c); its own train/evaluate/compare subcommands
are dropped, since evaluation and statistics now live in steps 03 and 09.

Every setting not passed on the command line comes from --config (default
configs/rl_final.yaml), which is the paper's single frozen, cited configuration. Passing a
flag overrides that file for this run only; the file itself is never modified by this script.

Ground truth is NOT read during training (contexts are sampled on the fly each episode);
this matches PPO training does not read ground-truth files at all in the audit.

A finished run (--out has a meta.json with status "complete") is skipped by default, so the
whole 2a/2b/2c shell loop can be re-run after a Colab disconnect without tracking by hand
what already finished; pass --force to discard it and start over from scratch, or --resume to
continue an interrupted one.

Usage:
    python pipeline/02_train_agent.py --algo ppo --reward_mode feasibility_gated --seed 42 \
        --out runs/2a_reward_mode/feasibility_gated/seed42
    python pipeline/02_train_agent.py --algo sac --reward_mode feasibility_gated --seed 42 \
        --out runs/2b_algo/sac/seed42
    python pipeline/02_train_agent.py --out runs/.../seed42 --resume   # continue an interrupted run
    python pipeline/02_train_agent.py --out runs/.../seed42 --force    # discard and restart
    python pipeline/02_train_agent.py --out runs/.../seed42 --set n_epochs=4   # one-off override
    python pipeline/02_train_agent.py --algo ppo --reward_mode lagrangian --seed 42 \
        --out /tmp/check --dry_run   # print the resolved config only, no torch/sb3 import

Smoke test (~1-2 min, no GPU needed):
    python pipeline/02_train_agent.py --algo ppo --reward_mode feasibility_gated --seed 0 \
        --timesteps 2048 --n_envs 2 --checkpoint_every 1024 --out /tmp/train_smoke
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hssbeamgen.train_utils import (  # noqa: E402
    apply_overrides, base_config_from_meta, build_callbacks, build_model, build_vec_env,
    cfg_for_json, load_resume_state, restore_lagrangian_callback, resolve_config,
    save_resume_state, wrap_normalize,
)

REWARD_MODES = ("feasibility_gated", "lagrangian", "shaped")
# Baked into the saved model / VecNormalize / replay-buffer shapes: cannot change on --resume.
SHAPE_LOCKED_ON_RESUME = ("n_envs", "net_arch")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_meta(run_dir, cfg, args, extra):
    import hssbeamgen
    env_file = os.path.join(os.path.dirname(hssbeamgen.__file__), "envs", "hss_env.py")
    meta = dict(
        script="pipeline/02_train_agent.py", args=vars(args), config=cfg_for_json(cfg),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        sha256=dict(hss_env=_sha256(env_file)), python=platform.python_version(), **extra,
    )
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run directory (checkpoints, meta.json, tensorboard/)")
    p.add_argument("--algo", choices=("ppo", "sac", "td3", "ddpg"))
    p.add_argument("--reward_mode", choices=REWARD_MODES)
    p.add_argument("--seed", type=int)
    p.add_argument("--config", default="configs/rl_final.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override one frozen-config value for this run only")
    p.add_argument("--resume", action="store_true", help="continue from this --out directory")
    p.add_argument("--force", action="store_true",
                   help="discard an existing (complete or partial) run in --out and start over")
    p.add_argument("--dry_run", action="store_true",
                   help="print the fully resolved config as JSON and exit; imports no torch/sb3")
    p.add_argument("--no_tensorboard", action="store_true")
    # Direct overrides for anything commonly varied without editing the config file:
    for k, t in [("timesteps", int), ("n_envs", int), ("checkpoint_every", int),
                 ("economy_metric", str), ("economy_reward_mode", str), ("lr", float),
                 ("log_std_anneal", lambda v: v.lower() == "true")]:
        p.add_argument(f"--{k}", type=t, default=None)
    args = p.parse_args()

    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive.")

    existing_meta_path = os.path.join(args.out, "meta.json")
    existing = json.load(open(existing_meta_path)) if os.path.exists(existing_meta_path) else None

    if args.force and existing is not None:
        print(f"--force: removing existing run in {args.out} (status was {existing.get('status')!r})")
        shutil.rmtree(args.out)
        existing = None

    if not args.resume and not args.force and existing is not None:
        if existing.get("status") == "complete":
            print(f"{args.out} already complete ({existing.get('final_num_timesteps'):,} steps); "
                 f"skipping (pass --force to redo, or --resume if it was actually interrupted).")
            return
        raise SystemExit(f"{args.out} has an incomplete run (status={existing.get('status')!r}); "
                         f"pass --resume to continue it or --force to discard and restart.")

    resuming = args.resume
    prev = existing if resuming else None
    if resuming:
        if prev is None:
            raise SystemExit(f"--resume given but no meta.json in {args.out}; nothing to resume.")
        for k in ("algo", "reward_mode", "seed"):
            if getattr(args, k) is None:
                setattr(args, k, prev["args"][k])
            elif getattr(args, k) != prev["args"][k]:
                raise SystemExit(f"--{k}={getattr(args, k)} conflicts with the original run's "
                                 f"{k}={prev['args'][k]!r}; omit --{k} to resume as-is.")
        for k in SHAPE_LOCKED_ON_RESUME:
            if getattr(args, k, None) is not None and getattr(args, k) != prev["config"][k]:
                raise SystemExit(f"--{k} cannot change on --resume (baked into the saved model/"
                                 f"VecNormalize shapes); original run used {k}={prev['config'][k]!r}.")
        print(f"resuming: algo={args.algo} reward_mode={args.reward_mode} seed={args.seed}")
    else:
        missing = [k for k in ("algo", "reward_mode", "seed") if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"--{missing[0]} is required for a new run (only optional with --resume).")

    overrides = {k: getattr(args, k) for k in
                ("timesteps", "n_envs", "checkpoint_every", "economy_metric",
                 "economy_reward_mode", "lr", "log_std_anneal")}
    if resuming:
        # Base = what the run actually trained with (meta.json), NOT the config file's current
        # defaults -- e.g. --n_envs 2 on the original run must not silently become the file's 8.
        cfg = apply_overrides(base_config_from_meta(prev), args.algo, overrides=overrides, sets=args.set)
    else:
        cfg = resolve_config(args.config, args.algo, overrides=overrides, sets=args.set)

    if args.dry_run:
        print(json.dumps(dict(algo=args.algo, reward_mode=args.reward_mode, seed=args.seed,
                              resuming=resuming, out=args.out, config=cfg), indent=2))
        return

    os.makedirs(args.out, exist_ok=True)
    tb_dir = None if args.no_tensorboard else os.path.join(args.out, "tensorboard")

    vec_env = build_vec_env(cfg, args.reward_mode, args.seed)
    vec_env = wrap_normalize(vec_env, cfg)
    callbacks, lag_cb = build_callbacks(cfg, args.reward_mode, args.out)

    resumed_from = 0
    if resuming:
        loaded = load_resume_state(args.out, cfg, vec_env)
        if loaded is None:
            raise SystemExit(f"--resume given but no resume_state.json in {args.out} yet "
                             f"(first checkpoint_every={cfg['checkpoint_every']} steps not reached).")
        model, vec_env, state = loaded
        restore_lagrangian_callback(lag_cb, state)
        resumed_from = state["num_timesteps"]
        # callbacks read training_env as a property off model.get_env(), which .load(env=vec_env)
        # above already set to the resumed vec_env -- nothing to rebind here.
        print(f"resumed at {resumed_from:,} / {cfg['timesteps']:,} steps")
    else:
        model = build_model(cfg, vec_env, args.seed, tb_dir)

    remaining = cfg["timesteps"] - resumed_from
    if remaining <= 0:
        print(f"already at {resumed_from:,} >= {cfg['timesteps']:,} target steps; nothing to do.")
        return
    print(f"algo={args.algo} reward_mode={args.reward_mode} seed={args.seed} "
         f"-> training {remaining:,} steps ({resumed_from:,} already done) in {args.out}")

    write_meta(args.out, cfg,
              argparse.Namespace(algo=args.algo, reward_mode=args.reward_mode, seed=args.seed),
              extra=dict(status="running", resumed_from_steps=resumed_from))
    t0 = time.time()
    model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=not resuming,
               progress_bar=False)
    save_resume_state(model, vec_env, args.out, cfg, lag_cb)
    model.save(os.path.join(args.out, "final_model"))
    vec_env.save(os.path.join(args.out, "final_vecnormalize.pkl"))

    write_meta(args.out, cfg,
              argparse.Namespace(algo=args.algo, reward_mode=args.reward_mode, seed=args.seed),
              extra=dict(status="complete", resumed_from_steps=resumed_from,
                         final_num_timesteps=int(model.num_timesteps),
                         wall_time_min=round((time.time() - t0) / 60, 2)))
    print(f"done: {model.num_timesteps:,} steps, {(time.time()-t0)/60:.1f} min -> {args.out}/final_model.zip")


if __name__ == "__main__":
    main()
