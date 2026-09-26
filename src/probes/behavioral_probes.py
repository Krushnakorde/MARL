"""
Frozen behavioral-probe module.

Rather than judging individuality from raw policy-distance metrics collected
during training, this module freezes an agent's weights at a checkpoint,
disables learning, and evaluates it across standardized, repeated trials in
fixed environment configurations (synopsis: "agents are frozen at regular
intervals and evaluated across standardized behavioral assays").
"""
import numpy as np

from ..environment.grid_world import GridWorld

FEATURE_NAMES = [
    "resources_collected", "unique_cells_visited", "action_entropy",
    "move_up_frac", "move_down_frac", "move_left_frac", "move_right_frac",
    "stay_frac", "near_others_frac",
]


def _behavior_features(env, agent_idx):
    total_steps = max(1, len(env.action_log[agent_idx]))
    counts = np.bincount(env.action_log[agent_idx], minlength=5).astype(np.float32)
    fracs = counts / total_steps
    nonzero = fracs[fracs > 0]
    entropy = float(-(nonzero * np.log(nonzero + 1e-8)).sum())
    return np.array([
        env.cumulative_resources[agent_idx],
        len(env.visited[agent_idx]),
        entropy,
        fracs[1], fracs[2], fracs[3], fracs[4], fracs[0],
        env.near_others_steps[agent_idx] / total_steps,
    ], dtype=np.float32)


def run_behavioral_probe(agent_templates, checkpoint_states, env_config,
                          num_trials=10, base_seed=1000, deterministic=False):
    """
    agent_templates: list of Agent objects (architecture only -- weights get
        overwritten from checkpoint_states before probing).
    checkpoint_states: list of per-agent frozen state_dicts to evaluate.
    env_config: kwargs defining the standardized test context (GridWorld).
    Returns: (profiles, feature_names) where profiles has shape
        (num_agents, num_trials, num_features).
    """
    num_agents = len(agent_templates)
    for agent, state in zip(agent_templates, checkpoint_states):
        agent.load_frozen(state)

    profiles = np.zeros((num_agents, num_trials, len(FEATURE_NAMES)), dtype=np.float32)

    for trial in range(num_trials):
        seed = base_seed + trial
        rng = np.random.default_rng(seed)
        env = GridWorld(rng=np.random.default_rng(seed), **env_config)
        obs = env.reset(seed=seed)
        for agent in agent_templates:
            agent.reset_memory()
        done = False
        while not done:
            actions = []
            for i, agent in enumerate(agent_templates):
                a, _, _, _, _ = agent.act(obs[i], rng, deterministic=deterministic)
                actions.append(a)
            obs, rewards, done, info = env.step(actions)
        for i in range(num_agents):
            profiles[i, trial] = _behavior_features(env, i)

    return profiles, FEATURE_NAMES
