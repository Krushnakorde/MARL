"""
Policy and value module.

A small actor-critic network implemented directly in NumPy, with manual
forward and backward passes (no autodiff framework required -- this
environment has no internet access to install PyTorch / Stable-Baselines3).
Trained via REINFORCE with a learned value baseline (advantage = return -
value), i.e. a simple one-hidden-layer decentralized actor-critic.
"""
import numpy as np


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)


class MLPPolicyValue:
    def __init__(self, obs_dim, action_dim, hidden_dim=32, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.params = {
            "W1": rng.normal(0, scale1, size=(obs_dim, hidden_dim)).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "Wpi": rng.normal(0, scale2, size=(hidden_dim, action_dim)).astype(np.float32),
            "bpi": np.zeros(action_dim, dtype=np.float32),
            "Wv": rng.normal(0, scale2, size=(hidden_dim, 1)).astype(np.float32),
            "bv": np.zeros(1, dtype=np.float32),
        }
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def state_dict(self):
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state):
        self.params = {k: v.copy() for k, v in state.items()}

    def forward(self, obs):
        z1 = obs @ self.params["W1"] + self.params["b1"]
        h = np.tanh(z1)
        logits = h @ self.params["Wpi"] + self.params["bpi"]
        probs = softmax(logits)
        value = float((h @ self.params["Wv"] + self.params["bv"])[0])
        cache = {"obs": obs, "h": h, "probs": probs}
        return probs, value, cache

    def act(self, obs, rng, deterministic=False):
        probs, value, cache = self.forward(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(self.action_dim, p=probs))
        logprob = float(np.log(probs[action] + 1e-8))
        return action, logprob, value, probs, cache

    def compute_grads(self, cache, action, advantage, value, target_return):
        """Gradients of: -advantage * log pi(a|s)  [policy]
                        + 0.5 * (value - target_return)^2  [value baseline]
        with respect to every parameter, via manual backprop through the
        single tanh hidden layer."""
        obs, h, probs = cache["obs"], cache["h"], cache["probs"]

        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action] = 1.0
        dlogits = advantage * (probs - onehot)

        dWpi = np.outer(h, dlogits)
        dbpi = dlogits
        dh_pi = dlogits @ self.params["Wpi"].T

        dvalue = (value - target_return)
        dWv = h.reshape(-1, 1) * dvalue
        dbv = np.array([dvalue], dtype=np.float32)
        dh_v = self.params["Wv"].flatten() * dvalue

        dh = dh_pi + dh_v
        dz1 = dh * (1.0 - h ** 2)
        dW1 = np.outer(obs, dz1)
        db1 = dz1

        return {"W1": dW1, "b1": db1, "Wpi": dWpi, "bpi": dbpi, "Wv": dWv, "bv": dbv}

    def apply_grads(self, grads, lr=0.01, clip=5.0):
        for k in self.params:
            g = np.clip(grads[k], -clip, clip)
            self.params[k] -= lr * g
