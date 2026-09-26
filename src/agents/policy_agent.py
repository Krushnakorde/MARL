"""
Agent perception + memory module.

Wraps a policy/value network with a short-term perceptual memory buffer, and
implements the decentralized, independent-learning training update for one
agent (each agent updates only from its own private experience -- see
synopsis Section 4, System Overview).

policy_type='feedforward': the agent acts on the current raw observation only.
policy_type='recurrent': the agent acts on a sliding window of the last
    `memory_length` raw observations, concatenated together. This is a
    deliberate, dependency-free stand-in for a true recurrent (LSTM) policy --
    it gives the agent access to short-term history without requiring
    backprop-through-time in hand-rolled NumPy. See README for how to swap in
    a real LSTM if a deep learning framework becomes available.
"""
from collections import deque

import numpy as np

from .networks import MLPPolicyValue


class Agent:
    def __init__(self, raw_obs_dim, action_dim, policy_type="feedforward",
                 memory_length=4, hidden_dim=32, seed=0, gamma=0.97, lr=0.02):
        self.policy_type = policy_type
        self.memory_length = memory_length if policy_type == "recurrent" else 1
        self.raw_obs_dim = raw_obs_dim
        self.effective_obs_dim = raw_obs_dim * self.memory_length
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.net = MLPPolicyValue(self.effective_obs_dim, action_dim, hidden_dim, seed=seed)
        self._history = deque(maxlen=self.memory_length)
        self.reset_memory()

    def reset_memory(self):
        self._history.clear()
        for _ in range(self.memory_length):
            self._history.append(np.zeros(self.raw_obs_dim, dtype=np.float32))

    def _encode(self, raw_obs):
        self._history.append(raw_obs)
        return np.concatenate(list(self._history)).astype(np.float32)

    def peek_encode(self, raw_obs):
        """Encode without mutating memory -- used for standardized probe
        batches where we want to evaluate each fixed observation independently."""
        hist = list(self._history)[1:] + [raw_obs]
        return np.concatenate(hist).astype(np.float32)

    def act(self, raw_obs, rng, deterministic=False):
        enc = self._encode(raw_obs)
        return self.net.act(enc, rng, deterministic=deterministic)

    def train_on_episode(self, trajectory):
        """trajectory: list of dicts with keys action, reward, value, cache
        (one entry per timestep of one episode). Performs a Monte-Carlo
        REINFORCE + baseline update, averaged over the episode."""
        returns = np.zeros(len(trajectory), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(trajectory))):
            running = trajectory[t]["reward"] + self.gamma * running
            returns[t] = running

        if len(returns) > 1 and returns.std() > 1e-6:
            norm_returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        else:
            norm_returns = returns

        accum = None
        for t, step in enumerate(trajectory):
            advantage = norm_returns[t] - step["value"]
            grads = self.net.compute_grads(step["cache"], step["action"], advantage,
                                            step["value"], norm_returns[t])
            if accum is None:
                accum = {k: v.copy() for k, v in grads.items()}
            else:
                for k in accum:
                    accum[k] += grads[k]
        n = len(trajectory)
        for k in accum:
            accum[k] /= n
        self.net.apply_grads(accum, lr=self.lr)

    def clone_frozen(self):
        return self.net.state_dict()

    def load_frozen(self, state):
        self.net.load_state_dict(state)
