"""
Environment module.

A partially observable, multi-agent grid world used to study emergent
behavioral individuality in cloned agents (see synopsis Section 6, Step 1-2).

- Agents move on a bounded grid_size x grid_size grid (no wraparound).
- Resources spawn on empty cells and respawn after a delay once collected.
- Each agent perceives only a local window around itself (partial
  observability), split into three channels: resources, other agents
  (visibility gated by the social interaction topology), and walls /
  out-of-bounds cells.
- Reward is +1 for the agent that collects a resource; ties are broken by
  agent index (first claim wins).
"""
import numpy as np

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


class GridWorld:
    def __init__(self, grid_size=10, num_agents=4, resource_density=0.12,
                 view_radius=2, respawn_delay=5, interaction_topology="fully_mixed",
                 local_radius=3, max_steps=60, rng=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.resource_density = resource_density
        self.view_radius = view_radius
        self.respawn_delay = respawn_delay
        self.interaction_topology = interaction_topology  # fully_mixed | local | isolated
        self.local_radius = local_radius
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()

        self.window = 2 * view_radius + 1
        # 3 channels (resources, other agents, walls) over the local window,
        # plus own normalized (row, col) position and the last reward received.
        self.obs_dim = 3 * self.window * self.window + 3
        self.action_dim = len(ACTION_DELTAS)

        # populated by reset()
        self.t = 0
        self.resources = None
        self.respawn_timer = None
        self.positions = None
        self.last_reward = None
        self.cumulative_resources = None
        self.visited = None
        self.action_log = None
        self.near_others_steps = None

    def reset(self, spawn_positions=None, resource_seed_density=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        density = self.resource_density if resource_seed_density is None else resource_seed_density

        self.t = 0
        self.resources = (self.rng.random((self.grid_size, self.grid_size)) < density).astype(np.float32)
        self.respawn_timer = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        if spawn_positions is not None:
            self.positions = np.array(spawn_positions, dtype=np.int64).copy()
        else:
            self.positions = self.rng.integers(0, self.grid_size, size=(self.num_agents, 2))

        self.resources[self.positions[:, 0], self.positions[:, 1]] = 0.0
        self.last_reward = np.zeros(self.num_agents, dtype=np.float32)
        self.cumulative_resources = np.zeros(self.num_agents, dtype=np.float32)
        self.visited = [set() for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.visited[i].add(tuple(int(x) for x in self.positions[i]))
        self.action_log = [[] for _ in range(self.num_agents)]
        self.near_others_steps = np.zeros(self.num_agents, dtype=np.float32)

        return self._get_obs_all()

    def _visible_agent_mask(self, agent_idx):
        """Which other agents are perceivable by agent_idx, per the social
        interaction topology (Social interaction module)."""
        if self.interaction_topology == "isolated":
            return np.zeros(self.num_agents, dtype=bool)
        mask = np.ones(self.num_agents, dtype=bool)
        mask[agent_idx] = False
        if self.interaction_topology == "fully_mixed":
            return mask
        if self.interaction_topology == "local":
            d = np.abs(self.positions - self.positions[agent_idx]).sum(axis=1)
            return mask & (d <= self.local_radius)
        return mask

    def _get_obs(self, agent_idx):
        r, c = self.positions[agent_idx]
        w = self.view_radius
        size = self.window
        res_ch = np.zeros((size, size), dtype=np.float32)
        agent_ch = np.zeros((size, size), dtype=np.float32)
        wall_ch = np.zeros((size, size), dtype=np.float32)

        visible = self._visible_agent_mask(agent_idx)
        other_positions = self.positions[visible]

        for dr in range(-w, w + 1):
            for dc in range(-w, w + 1):
                rr, cc = r + dr, c + dc
                ri, ci = dr + w, dc + w
                if rr < 0 or rr >= self.grid_size or cc < 0 or cc >= self.grid_size:
                    wall_ch[ri, ci] = 1.0
                else:
                    res_ch[ri, ci] = self.resources[rr, cc]
        for (orow, ocol) in other_positions:
            dr, dc = int(orow) - int(r), int(ocol) - int(c)
            if -w <= dr <= w and -w <= dc <= w:
                agent_ch[dr + w, dc + w] = 1.0

        flat = np.concatenate([res_ch.flatten(), agent_ch.flatten(), wall_ch.flatten()])
        own_pos_norm = self.positions[agent_idx].astype(np.float32) / max(1, self.grid_size - 1)
        extra = np.array([own_pos_norm[0], own_pos_norm[1], self.last_reward[agent_idx]], dtype=np.float32)
        return np.concatenate([flat, extra]).astype(np.float32)

    def _get_obs_all(self):
        return [self._get_obs(i) for i in range(self.num_agents)]

    def step(self, actions):
        """Advance the environment by one timestep given simultaneous actions,
        one per agent. Returns (obs_list, rewards, done, info)."""
        self.t += 1
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        new_positions = self.positions.copy()
        for i, a in enumerate(actions):
            dr, dc = ACTION_DELTAS[int(a)]
            nr = int(np.clip(self.positions[i, 0] + dr, 0, self.grid_size - 1))
            nc = int(np.clip(self.positions[i, 1] + dc, 0, self.grid_size - 1))
            new_positions[i] = (nr, nc)
            self.action_log[i].append(int(a))

        self.positions = new_positions

        # Resource collection: first agent (by index) to occupy a resourced
        # cell this step claims it.
        claimed = set()
        for i in range(self.num_agents):
            pos = (int(self.positions[i, 0]), int(self.positions[i, 1]))
            if pos not in claimed and self.resources[pos] > 0:
                rewards[i] += 1.0
                self.resources[pos] = 0.0
                self.respawn_timer[pos] = self.respawn_delay
                self.cumulative_resources[i] += 1.0
                claimed.add(pos)
            self.visited[i].add(pos)

        # "Near others" tracking is independent of the observation topology --
        # it is a ground-truth behavioral signal used only for analysis.
        for i in range(self.num_agents):
            d = np.abs(self.positions - self.positions[i]).sum(axis=1)
            d[i] = 999
            if (d <= 1).any():
                self.near_others_steps[i] += 1.0

        expiring = (self.respawn_timer == 1)
        if expiring.any():
            free = expiring & (self.rng.random(expiring.shape) < 0.5)
            self.resources[free] = 1.0
        self.respawn_timer[self.respawn_timer > 0] -= 1

        self.last_reward = rewards
        done = self.t >= self.max_steps
        obs = self._get_obs_all()
        info = {"positions": self.positions.copy()}
        return obs, rewards, done, info
