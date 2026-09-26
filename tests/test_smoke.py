"""
Smoke test: runs a tiny end-to-end experiment (training + checkpointing +
frozen behavioral probing + metrics) and checks basic invariants. Not a full
unit test suite, but a fast way to confirm the whole pipeline actually runs.

Usage:
    python tests/test_smoke.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.experiment.run_experiment import run  # noqa: E402


def main():
    config = {
        "experiment_name": "smoke_test",
        "seed": 0,
        "init_seed": 0,
        "grid_size": 6,
        "num_agents": 2,
        "resource_density": 0.15,
        "view_radius": 2,
        "respawn_delay": 4,
        "interaction_topology": "fully_mixed",
        "local_radius": 3,
        "max_steps": 15,
        "policy_type": "feedforward",
        "memory_length": 3,
        "hidden_dim": 16,
        "gamma": 0.95,
        "learning_rate": 0.02,
        "num_episodes": 6,
        "checkpoint_interval": 3,
        "early_phase_episodes": 2,
        "early_resource_density": 0.3,
        "spawn_mode": "quadrant",
        "num_probe_trials": 3,
    }
    out_dir = "experiments_smoke_tmp"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    exp_dir, summary = run(config, base_dir=out_dir)

    assert os.path.exists(os.path.join(exp_dir, "summary.json"))
    assert os.path.exists(os.path.join(exp_dir, "checkpoint_metrics.csv"))
    assert os.path.exists(os.path.join(exp_dir, "episode_log.csv"))
    assert summary["num_checkpoints"] >= 1

    print("SMOKE TEST PASSED")
    shutil.rmtree(out_dir)


if __name__ == "__main__":
    main()
