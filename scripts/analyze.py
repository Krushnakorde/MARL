"""
Generate plots and a markdown report from a finished experiment directory.

Usage:
    python scripts/analyze.py --experiment experiments/<experiment_id>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate plots and a summary report for an experiment.")
    parser.add_argument("--experiment", type=str, required=True, help="Path to an experiment output directory.")
    args = parser.parse_args()

    ckpt_df = pd.read_csv(os.path.join(args.experiment, "checkpoint_metrics.csv"))
    episode_df = pd.read_csv(os.path.join(args.experiment, "episode_log.csv"))
    with open(os.path.join(args.experiment, "summary.json")) as f:
        summary = json.load(f)

    plots_dir = os.path.join(args.experiment, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Policy divergence & behavioral repeatability across checkpoints
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ckpt_df["checkpoint"], ckpt_df["mean_policy_divergence"], marker="o",
              color="#2563eb", label="Policy divergence (KL)")
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Mean policy divergence", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(ckpt_df["checkpoint"], ckpt_df["mean_repeatability_icc"], marker="s",
              color="#16a34a", label="Repeatability (ICC)")
    ax2.set_ylabel("Mean behavioral repeatability (ICC)", color="#16a34a")
    ax2.set_ylim(0, 1)
    fig.suptitle(f"{summary['experiment_name']}: divergence & repeatability over training")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "divergence_repeatability.png"), dpi=150)
    plt.close(fig)

    # 2. Cross-context consistency
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ckpt_df["checkpoint"], ckpt_df["cross_context_consistency"], marker="^", color="#d97706")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Cross-context consistency (Spearman rho)")
    ax.set_title(f"{summary['experiment_name']}: cross-context behavioral consistency")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cross_context_consistency.png"), dpi=150)
    plt.close(fig)

    # 3. Per-agent training reward trajectories
    agent_cols = [c for c in episode_df.columns if c.startswith("agent_") and c.endswith("_reward")]
    fig, ax = plt.subplots(figsize=(7, 4))
    window = max(1, len(episode_df) // 20)
    for col in agent_cols:
        smoothed = episode_df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(episode_df["episode"], smoothed, label=col.replace("_reward", ""))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title(f"{summary['experiment_name']}: per-agent training reward")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "reward_trajectories.png"), dpi=150)
    plt.close(fig)

    stability_line = (
        f"- Mean temporal stability (across checkpoints): {summary['mean_temporal_stability']:.3f}"
        if summary["mean_temporal_stability"] is not None
        else "- Mean temporal stability: N/A (needs at least 2 checkpoints)"
    )
    topo = summary["final_topology_summary"]
    report_lines = [
        f"# Experiment Report: {summary['experiment_name']}",
        "",
        f"- Checkpoints evaluated: {summary['num_checkpoints']}",
        stability_line,
        f"- Final interaction topology: {topo['topology']} "
        f"(avg degree {topo['avg_degree']:.2f}, density {topo['density']:.2f})",
        "",
        "## Final checkpoint metrics",
        "",
        ckpt_df.tail(1).to_string(index=False) if not ckpt_df.empty else "N/A",
        "",
        "Plots saved under `plots/`: divergence_repeatability.png, "
        "cross_context_consistency.png, reward_trajectories.png.",
    ]
    with open(os.path.join(args.experiment, "REPORT.md"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"Plots written to: {plots_dir}")
    print(f"Report written to: {os.path.join(args.experiment, 'REPORT.md')}")


if __name__ == "__main__":
    main()
