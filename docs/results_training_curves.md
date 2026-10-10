# Training curves (supplement)

Data: `results/training_curves/` (40 complete runs: 2a 15, 2b 15 without PPO, 2c 10; `index.csv` lists run, last step,
commit, `git_dirty`, resumes). Table: `paper/tables/t13_training_summary.*`. Figures: `figures/fig7_training_curves.*` (objectives, algorithms), `figures/fig8_reward_modes.*` (reward modes, Lagrangian),
`figures/fig7_8_combined_training_curves.*` (one page).
Exported by `pipeline/13_export_training_curves.py`. Only training-episode quantities exist, not intermediate evaluations, so no
curve of gap against training steps is available. Rewards are per-episode sums on the scale of their reward mode; the
comparison metric of the paper is the gap.

## What the curves show (training reward, 1M steps, all arms the same budget)
- **Episode length.** Over the last 100k steps PPO runs 38–39 of the 40 steps per episode, DDPG 36; SAC runs 12 and TD3 20
  (t13). Their episode reward is a sum over fewer steps and is not comparable with PPO's (Fig. 7b, c).
- **Late-training reward.** The end-window reward is at least twice as negative as the mid-window reward in 4 of 10 SAC / TD3
  seeds (SAC 1, TD3 3) and in none of the PPO or DDPG seeds. The algorithm comparison (2b) evaluates the final checkpoint at
  1M steps; there is no checkpoint selection.
- **Not plateaued at 1M.** Mean reward of the PPO `feasibility_gated` runs over 0.7–0.8M, 0.8–0.9M and 0.9–1.0M is
  −50.7, −46.9, −44.1; Lagrangian −134.8, −131.2, −129.9. The `shaped` runs are still rising steeply at 1M (two of five seeds
  have positive reward after 0.9M). Longer training was not tried.
- **Lagrangian runs.** Mean constraint violations fall during training for all three constraints; the multipliers are still
  slowly increasing at 1M (Fig. 8c, d).

## Audit trail
- All 40 runs have `status = complete`, one event file, `resumed_from_steps = 0`, `git_dirty = False`.
- `2b_algorithm__ddpg__seed42` was trained at commit `9bb39c1`, seeds 43–46 at `e1a0afa`; the diff between the two commits
  in the training code and config is documentation text only (a comment path in `rl_final.yaml`, a docstring in
  `log_std_anneal.py`).
