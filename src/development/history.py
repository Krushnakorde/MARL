"""
Developmental-history module.

Controls early-phase developmental experience while keeping the reward
function and task objective fixed for the entire run (synopsis Section 6,
Step 3: "Introduce different early experiences while keeping the
architecture, initial parameters, reward function, task objective, and
training budget constant.").

Supported manipulations:
  - early_phase_episodes: how many initial episodes count as "development".
  - early_resource_density: an optionally different resource density used
    only during the early phase (e.g. a "deprived" vs "enriched" start).
  - spawn_mode="quadrant": during the early phase, agent i is confined to a
    fixed grid quadrant (a controlled, divergent early spatial experience per
    agent); after the early phase every agent spawns uniformly at random,
    identically to every other agent.
"""
import numpy as np


class DevelopmentalHistory:
    def __init__(self, num_agents, grid_size, early_phase_episodes=20,
                 early_resource_density=None, spawn_mode="quadrant", rng=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.early_phase_episodes = early_phase_episodes
        self.early_resource_density = early_resource_density
        self.spawn_mode = spawn_mode
        self.rng = rng if rng is not None else np.random.default_rng()
        self._quadrants = self._assign_quadrants()

    def _assign_quadrants(self):
        half = max(1, self.grid_size // 2)
        bounds = [(0, half), (half, self.grid_size)]
        combos = [(r, c) for r in bounds for c in bounds]
        return [combos[i % len(combos)] for i in range(self.num_agents)]

    def is_early_phase(self, episode_idx):
        return episode_idx < self.early_phase_episodes

    def spawn_positions(self, episode_idx):
        if not self.is_early_phase(episode_idx) or self.spawn_mode != "quadrant":
            return self.rng.integers(0, self.grid_size, size=(self.num_agents, 2))
        positions = np.zeros((self.num_agents, 2), dtype=np.int64)
        for i in range(self.num_agents):
            (r0, r1), (c0, c1) = self._quadrants[i]
            positions[i, 0] = self.rng.integers(r0, max(r0 + 1, r1))
            positions[i, 1] = self.rng.integers(c0, max(c0 + 1, c1))
        return positions

    def resource_density(self, episode_idx, base_density):
        if self.is_early_phase(episode_idx) and self.early_resource_density is not None:
            return self.early_resource_density
        return base_density
