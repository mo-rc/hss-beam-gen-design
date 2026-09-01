#!/usr/bin/env bash
# Cleaner version of the seed42 extension: holds the log_std ceiling
# PINNED at ~1.0 for the entire resumed chunk (ceiling_start=1.0001,
# ceiling_end=1.0 -- interpolates to an effectively constant ~1.0 ceiling
# regardless of progress fraction, satisfying the class's
# ceiling_end < ceiling_start assertion without giving the policy any
# renewed headroom to raise std again). This is the pure test of "does
# seed42's policy just need more updates at the SAME low noise level it
# already reached, with nothing else changed."

python research/scripts/resume_training.py \
  --reward_mode feasibility_gated --economy_metric cost \
  --run_name gated_cost_merged_seed42 --seed 42 \
  --total_timesteps 1300000 --chunk_timesteps 300000 --n_envs 8 \
  --economy_reward_mode log_relative \
  --log_std_anneal --log_std_ceiling_start 1.0001 --log_std_ceiling_end 1.0 \
  --log_std_anneal_start_frac 0.0 \
  --out_dir research/models

python research/scripts/evaluate.py \
  --model_path research/models/gated_cost_merged_seed42/final_model \
  --algo ppo --economy_metric cost \
  --ground_truth_dir research/pretrain_data \
  --run_name gated_cost_merged_seed42_puresettle \
  --out_csv research/results/gated_cost_merged_seed42_puresettle_eval.csv
