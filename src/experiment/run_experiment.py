"""
Experiment orchestrator.

Ties together every module described in the synopsis's System Overview:
environment, agent perception/memory/policy, social interaction topology,
developmental history, frozen behavioral probes, behavioral analytics, and
experiment tracking.
"""
import numpy as np

from ..analytics.metrics import (cross_context_consistency, policy_divergence,
                                  repeatability_icc, temporal_stability)
from ..agents.policy_agent import Agent
from ..development.history import DevelopmentalHistory
from ..environment.grid_world import GridWorld
from ..probes.behavioral_probes import FEATURE_NAMES, run_behavioral_probe
from ..social.topology import topology_summary
from .tracker import ExperimentTracker


def build_env_config(config):
    return dict(
        grid_size=config["grid_size"],
        num_agents=config["num_agents"],
        resource_density=config["resource_density"],
        view_radius=config["view_radius"],
        respawn_delay=config["respawn_delay"],
        interaction_topology=config["interaction_topology"],
        local_radius=config["local_radius"],
        max_steps=config["max_steps"],
    )


def _make_agents(config, raw_obs_dim, action_dim, seed):
    return [
        Agent(raw_obs_dim, action_dim, policy_type=config["policy_type"],
              memory_length=config.get("memory_length", 4),
              hidden_dim=config["hidden_dim"], seed=seed,
              gamma=config["gamma"], lr=config["learning_rate"])
        for _ in range(config["num_agents"])
    ]


def run(config, base_dir="experiments", verbose=True):
    master_rng = np.random.default_rng(config["seed"])
    env_config = build_env_config(config)

    probe_env = GridWorld(**env_config)
    raw_obs_dim = probe_env.obs_dim
    action_dim = probe_env.action_dim

    # Cloned start: every agent begins from identical initial parameters.
    agents = _make_agents(config, raw_obs_dim, action_dim, seed=config["init_seed"])
    template_state = agents[0].net.state_dict()
    for agent in agents[1:]:
        agent.net.load_state_dict(template_state)

    history = DevelopmentalHistory(
        num_agents=config["num_agents"], grid_size=config["grid_size"],
        early_phase_episodes=config["early_phase_episodes"],
        early_resource_density=config.get("early_resource_density"),
        spawn_mode=config.get("spawn_mode", "quadrant"),
        rng=np.random.default_rng(config["seed"] + 1),
    )

    tracker = ExperimentTracker(base_dir, config["experiment_name"], config)
    env = GridWorld(**env_config, rng=np.random.default_rng(config["seed"] + 2))

    checkpoints = []  # (checkpoint_idx, episode_idx, [state_dicts])
    checkpoint_idx = 0

    for episode in range(config["num_episodes"]):
        spawn = history.spawn_positions(episode)
        density = history.resource_density(episode, config["resource_density"])
        obs = env.reset(spawn_positions=spawn, resource_seed_density=density)
        for agent in agents:
            agent.reset_memory()

        trajectories = [[] for _ in range(config["num_agents"])]
        episode_rewards = np.zeros(config["num_agents"], dtype=np.float32)
        done = False
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                a, logprob, value, probs, cache = agent.act(obs[i], master_rng)
                actions.append(a)
                trajectories[i].append({"action": a, "value": value, "cache": cache, "reward": None})
            obs, rewards, done, info = env.step(actions)
            for i in range(config["num_agents"]):
                trajectories[i][-1]["reward"] = float(rewards[i])
                episode_rewards[i] += rewards[i]

        for i, agent in enumerate(agents):
            agent.train_on_episode(trajectories[i])

        tracker.log_episode(episode, episode_rewards)

        is_last = episode == config["num_episodes"] - 1
        if (episode + 1) % config["checkpoint_interval"] == 0 or is_last:
            states = [a.clone_frozen() for a in agents]
            tracker.save_checkpoint(checkpoint_idx, episode, states)
            checkpoints.append((checkpoint_idx, episode, states))
            checkpoint_idx += 1
            if verbose:
                avg_r = float(episode_rewards.mean())
                print(f"[{config['experiment_name']}] episode {episode + 1}/{config['num_episodes']} "
                      f"checkpoint {checkpoint_idx} avg_reward={avg_r:.2f}")

    # --- Frozen behavioral probing across all saved checkpoints ---
    probe_agents = _make_agents(config, raw_obs_dim, action_dim, seed=0)

    context_a_config = dict(env_config)
    context_b_config = dict(env_config)
    context_b_config["resource_density"] = min(0.35, env_config["resource_density"] * 2.5)
    context_b_config["grid_size"] = max(6, env_config["grid_size"] - 2)

    profile_history_a = []
    standardized_obs_batch = None

    for ckpt_idx, episode_idx, states in checkpoints:
        profiles_a, _ = run_behavioral_probe(
            probe_agents, states, context_a_config,
            num_trials=config["num_probe_trials"], base_seed=5000 + ckpt_idx * 100)
        profiles_b, _ = run_behavioral_probe(
            probe_agents, states, context_b_config,
            num_trials=config["num_probe_trials"], base_seed=9000 + ckpt_idx * 100)

        for agent, state in zip(probe_agents, states):
            agent.load_frozen(state)
        if standardized_obs_batch is None:
            ref_env = GridWorld(**context_a_config, rng=np.random.default_rng(42))
            standardized_obs_batch = ref_env.reset(seed=42)
        div_matrix, mean_div = policy_divergence(probe_agents, standardized_obs_batch)

        icc_scores = repeatability_icc(profiles_a)
        _, cross_context_overall = cross_context_consistency(profiles_a, profiles_b)

        profile_history_a.append(profiles_a.mean(axis=1))

        metrics = {
            "mean_policy_divergence": mean_div,
            "mean_repeatability_icc": float(np.mean(icc_scores)),
            "cross_context_consistency": cross_context_overall,
        }
        tracker.log_checkpoint_metrics(ckpt_idx, episode_idx, metrics)

    per_agent_stability, mean_stability = temporal_stability(profile_history_a)

    final_positions = env.positions.tolist()
    topo_info = topology_summary(final_positions, config["interaction_topology"], config["local_radius"])

    summary = {
        "experiment_name": config["experiment_name"],
        "num_checkpoints": len(checkpoints),
        "mean_temporal_stability": mean_stability,
        "per_agent_temporal_stability": per_agent_stability.tolist() if per_agent_stability is not None else None,
        "final_topology_summary": topo_info,
        "feature_names": FEATURE_NAMES,
    }
    tracker.finalize(summary)
    return tracker.dir, summary
