"""
Experiment-tracking module.

Writes a reproducible, self-contained record of a run to disk:
  experiments/<experiment_id>/
    config.json              -- full run configuration
    checkpoints/*.npz         -- frozen per-agent weights at each checkpoint
    episode_log.csv           -- per-episode reward for every agent
    checkpoint_metrics.csv    -- behavioral metrics at each checkpoint
    summary.json              -- final summary statistics
"""
import json
import os
import time

import numpy as np
import pandas as pd


class ExperimentTracker:
    def __init__(self, base_dir, experiment_name, config):
        self.experiment_id = f"{experiment_name}_{int(time.time())}"
        self.dir = os.path.join(base_dir, self.experiment_id)
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "checkpoints"), exist_ok=True)
        self.config = config
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        self.checkpoint_rows = []
        self.episode_rows = []

    def log_episode(self, episode_idx, rewards_per_agent):
        row = {"episode": episode_idx}
        for i, r in enumerate(rewards_per_agent):
            row[f"agent_{i}_reward"] = float(r)
        self.episode_rows.append(row)

    def save_checkpoint(self, checkpoint_idx, episode_idx, agent_states):
        path = os.path.join(self.dir, "checkpoints", f"checkpoint_{checkpoint_idx:03d}.npz")
        flat = {}
        for i, state in enumerate(agent_states):
            for k, v in state.items():
                flat[f"agent{i}_{k}"] = v
        np.savez_compressed(path, **flat)
        return path

    def log_checkpoint_metrics(self, checkpoint_idx, episode_idx, metrics):
        row = {"checkpoint": checkpoint_idx, "episode": episode_idx}
        row.update(metrics)
        self.checkpoint_rows.append(row)

    def finalize(self, summary):
        pd.DataFrame(self.episode_rows).to_csv(os.path.join(self.dir, "episode_log.csv"), index=False)
        pd.DataFrame(self.checkpoint_rows).to_csv(os.path.join(self.dir, "checkpoint_metrics.csv"), index=False)
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=lambda o: float(o))
        print(f"Experiment saved to: {self.dir}")
