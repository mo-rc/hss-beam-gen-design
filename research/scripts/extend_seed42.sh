#!/usr/bin/env bash
# Extends the existing gated_cost_merged_seed42 checkpoint from 1.0M to
# 1.3M total steps.
#
# IMPORTANT -- what this actually does to the schedule (verified by tracing
# resume_training.py + log_std_anneal.py, not assumed): LogStdAnnealCallback
# is reconstructed on resume with total_timesteps=--total_timesteps (the NEW
# grand total, 1.3M). Since frac = num_timesteps / total_timesteps, at the
# moment this resumes (num_timesteps=1,000,000) frac = 1,000,000/1,300,000
# = 0.769, which recomputes the ceiling at that instant to ~2.6 std -- ABOVE
# seed42's actual converged std (~1.02). Because the callback only clamps a
# MAXIMUM (never raises std), this is a safe no-op at the exact resume point
# (std stays ~1.02, no backward jump) -- but it does give the policy renewed
# headroom to increase std again over the next chunk if the optimizer wants
# to, before the ceiling continues decaying down to 1.0 by the new 1.3M mark.
#
# So this is NOT a pure "same policy, just more time at the already-reached
# floor" test -- it's "resume with a slightly more gradual remaining anneal,
# reaching the floor 300k steps later." That's still a fair, informative
# test of the "seed42 needed more settling time" hypothesis, just not the
# purest possible version of it. If you want the pure version instead (hard
# floor held constant, zero renewed headroom), tell me and I'll write a
# small variant that passes ceiling_start=ceiling_end=1.0 for the resumed
# chunk instead of reusing the standard schedule.
#
# Do NOT change --seed, --reward_mode, --economy_metric,
# --economy_reward_mode, or --log_std_ceiling_start/end, or this stops being
# a controlled continuation of the same run.


python research/scripts/evaluate.py \
  --model_path research/models/gated_cost_merged_seed42/final_model \
  --algo ppo --economy_metric cost \
  --ground_truth_dir research/pretrain_data \
  --run_name gated_cost_merged_seed42_extended \
  --out_csv research/results/gated_cost_merged_seed42_extended_eval.csv
