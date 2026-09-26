"""
Behavioral analytics module.

Implements the four evaluation metrics from synopsis Section 7:
  - Policy Divergence: pairwise KL divergence between agents' action
    distributions on the same fixed (standardized) observations.
  - Behavioral Repeatability: an ICC(1)-style one-way random-effects
    intraclass correlation across repeated standardized trials, computed
    from first principles (no statsmodels dependency).
  - Temporal Stability: correlation of an agent's behavior profile between
    consecutive frozen checkpoints.
  - Cross-Context Consistency: rank correlation of agents' behavior profiles
    between two different standardized test environments.
"""
import numpy as np
from scipy.stats import spearmanr


def policy_divergence(agents, standardized_obs_batch):
    """agents: list of Agent objects with frozen weights already loaded.
    standardized_obs_batch: list of raw observation vectors, identical for
    every agent. Returns (n x n) mean-KL matrix and its off-diagonal mean."""
    n = len(agents)
    all_probs = []
    for agent in agents:
        agent.reset_memory()
        probs_list = [agent.net.forward(agent.peek_encode(raw_obs))[0] for raw_obs in standardized_obs_batch]
        all_probs.append(np.array(probs_list))

    div_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = all_probs[i] + 1e-8
            q = all_probs[j] + 1e-8
            kl = np.sum(p * np.log(p / q), axis=1)
            div_matrix[i, j] = float(np.mean(kl))
    mask = ~np.eye(n, dtype=bool)
    mean_divergence = float(div_matrix[mask].mean()) if n > 1 else 0.0
    return div_matrix, mean_divergence


def repeatability_icc(profiles):
    """One-way random-effects ICC(1): ICC = (MSB - MSW) / (MSB + (k-1)*MSW),
    where k = trials per agent. profiles: (num_agents, num_trials, num_features).
    Returns one ICC score per feature, clipped to [0, 1]."""
    n_agents, n_trials, n_features = profiles.shape
    scores = np.zeros(n_features, dtype=np.float32)
    for f in range(n_features):
        x = profiles[:, :, f]
        grand_mean = x.mean()
        agent_means = x.mean(axis=1)
        ssb = n_trials * np.sum((agent_means - grand_mean) ** 2)
        ssw = np.sum((x - agent_means[:, None]) ** 2)
        dof_b = max(1, n_agents - 1)
        dof_w = n_agents * (n_trials - 1)
        msb = ssb / dof_b
        msw = ssw / dof_w if dof_w > 0 else 1e-8
        icc = (msb - msw) / (msb + (n_trials - 1) * msw + 1e-8)
        scores[f] = float(np.clip(icc, 0.0, 1.0))
    return scores


def temporal_stability(profile_history):
    """profile_history: list over checkpoints of (num_agents, num_features)
    trial-averaged behavior profiles. Returns (per_agent_scores, mean_score)."""
    num_checkpoints = len(profile_history)
    if num_checkpoints < 2:
        return None, None
    num_agents = profile_history[0].shape[0]
    per_agent_scores = np.zeros(num_agents, dtype=np.float32)
    for i in range(num_agents):
        corrs = []
        for t in range(num_checkpoints - 1):
            a, b = profile_history[t][i], profile_history[t + 1][i]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                continue
            corrs.append(np.corrcoef(a, b)[0, 1])
        per_agent_scores[i] = float(np.mean(corrs)) if corrs else 0.0
    return per_agent_scores, float(np.mean(per_agent_scores))


def cross_context_consistency(profiles_a, profiles_b):
    """profiles_a, profiles_b: (num_agents, num_trials, num_features) from two
    different standardized test contexts. Returns (per_feature_corr, overall)
    using Spearman rank correlation of agents' trial-averaged scores."""
    mean_a = profiles_a.mean(axis=1)
    mean_b = profiles_b.mean(axis=1)
    num_features = mean_a.shape[1]

    per_feature_corr = np.zeros(num_features, dtype=np.float32)
    for f in range(num_features):
        if np.std(mean_a[:, f]) < 1e-8 or np.std(mean_b[:, f]) < 1e-8:
            continue
        rho, _ = spearmanr(mean_a[:, f], mean_b[:, f])
        per_feature_corr[f] = 0.0 if np.isnan(rho) else float(rho)

    primary_a, primary_b = mean_a[:, 0], mean_b[:, 0]
    if np.std(primary_a) < 1e-8 or np.std(primary_b) < 1e-8:
        overall = 0.0
    else:
        rho, _ = spearmanr(primary_a, primary_b)
        overall = 0.0 if np.isnan(rho) else float(rho)

    return per_feature_corr, overall
