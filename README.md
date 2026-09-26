# Emergence and Stability of Behavioral Individuality in Cloned MARL

A runnable, self-contained implementation of the experimental framework
described in the research synopsis "Emergence and Stability of Behavioral
Individuality in Cloned Multi-Agent Reinforcement Learning" (RPR-631-MCA).

This is a **real, decentralized multi-agent RL system** — agents are cloned
from identical initial parameters and independently learn via policy-gradient
training in a partially observable grid world. Their behavior is then frozen
at checkpoints and evaluated with the same statistical logic the synopsis
specifies (ICC repeatability, KL policy divergence, temporal stability,
cross-context consistency). Nothing in this codebase fabricates metrics —
every number in the output comes from an agent actually acting in the
environment.

## Why pure NumPy instead of PyTorch / Stable-Baselines3

The synopsis's software requirements list Stable-Baselines3/CleanRL and a
GPU. This code was generated in a sandboxed environment with **no internet
access**, so those packages cannot be installed. Instead, the actor-critic
network and its backward pass are implemented by hand in NumPy
(`src/agents/networks.py`) — a small one-hidden-layer network trained with
REINFORCE + a learned value baseline. It is fully functional, testable, and
produces genuine learning curves (see `tests/test_smoke.py` and the
validation run below), just smaller-scale than a GPU-accelerated deep RL
stack. If you later have PyTorch available, `MLPPolicyValue` is the only
class you'd need to swap out — every other module (environment, topology,
developmental history, probes, analytics, tracking) is framework-agnostic.

**Recurrent policy simplification:** true recurrent networks require
backpropagation-through-time, which is easy to get subtly wrong by hand.
`policy_type="recurrent"` instead gives the agent a sliding window of its
last `memory_length` raw observations (concatenated), a standard, honest
short-term-memory proxy that lets you compare "with memory" vs
"feedforward-only" agents without BPTT. See `src/agents/policy_agent.py`.

## How the code maps to the synopsis's System Overview (Section 6)

| Synopsis module | File |
|---|---|
| 1. Environment module | `src/environment/grid_world.py` |
| 2. Agent perception module | `src/agents/policy_agent.py` (`_encode`) |
| 3. Memory module | `src/agents/policy_agent.py` (`memory_length` window) |
| 4. Policy and value module | `src/agents/networks.py` |
| 5. Social interaction module | `src/social/topology.py` + `GridWorld._visible_agent_mask` |
| 6. Developmental-history module | `src/development/history.py` |
| 7. Frozen behavioral-probe module | `src/probes/behavioral_probes.py` |
| 8. Behavioral analytics module | `src/analytics/metrics.py` |
| 9. Experiment-tracking module | `src/experiment/tracker.py` |

`src/experiment/run_experiment.py` is the orchestrator that wires all nine
together into one training + evaluation run.

## Project layout

```
marl-individuality-platform/
├── requirements.txt
├── configs/
│   ├── default_experiment.json     # fully_mixed topology, feedforward policy
│   ├── isolated_feedforward.json   # isolated topology (no social signal)
│   └── local_recurrent.json        # local topology, memory-window policy
├── src/
│   ├── environment/grid_world.py
│   ├── agents/{networks.py, policy_agent.py}
│   ├── social/topology.py
│   ├── development/history.py
│   ├── probes/behavioral_probes.py
│   ├── analytics/metrics.py
│   └── experiment/{tracker.py, run_experiment.py}
├── scripts/
│   ├── train.py     # run an experiment end-to-end
│   └── analyze.py   # turn results into plots + a markdown report
└── tests/
    └── test_smoke.py  # fast end-to-end pipeline check
```

## Setup

```bash
cd marl-individuality-platform
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires only `numpy`, `pandas`, `matplotlib`, `scipy`, `networkx` — no
internet access needed beyond this one-time install.

## Running an experiment

```bash
python scripts/train.py --config configs/default_experiment.json
```

This trains `num_agents` cloned agents for `num_episodes` episodes,
checkpointing (freezing) their weights every `checkpoint_interval` episodes,
then runs frozen behavioral probes on every checkpoint in two standardized
test contexts and computes all four evaluation metrics. Everything is
written to `experiments/<experiment_name>_<timestamp>/`:

```
config.json               # exact config used (reproducibility)
checkpoints/*.npz         # frozen per-agent weights at each checkpoint
episode_log.csv           # per-episode reward, per agent
checkpoint_metrics.csv    # policy divergence / ICC / cross-context per checkpoint
summary.json              # temporal stability + final topology summary
```

Then generate plots and a short report:

```bash
python scripts/analyze.py --experiment experiments/<experiment_name>_<timestamp>
```

This adds a `plots/` folder (divergence & repeatability over training,
cross-context consistency, per-agent reward trajectories) and a
`REPORT.md` summary inside the experiment directory.

## Running the comparisons the synopsis asks for

The three provided configs let you run the causal comparisons from
Section 6, Step 6 directly:

```bash
python scripts/train.py --config configs/default_experiment.json     # fully_mixed
python scripts/train.py --config configs/isolated_feedforward.json   # isolated
python scripts/train.py --config configs/local_recurrent.json        # local + memory
```

Compare `checkpoint_metrics.csv` / `REPORT.md` across the three runs to see
how social topology and memory affect policy divergence, repeatability, and
cross-context consistency — exactly the comparison the synopsis's
`interactionTopology` / `policyType` ablations call for. Copy a config and
edit `num_agents`, `grid_size`, `early_phase_episodes`,
`early_resource_density`, or `spawn_mode` to design your own developmental-
history manipulations.

## Sanity-checking the pipeline

```bash
python tests/test_smoke.py
```

Runs a tiny (few-second) end-to-end experiment and asserts the whole
pipeline — training, checkpointing, probing, metrics, tracking — completes
without error. Useful after any code change.

## What the four metrics mean here (synopsis Section 7)

- **Policy divergence**: mean pairwise KL divergence between agents' action
  probability distributions, evaluated on the *same* fixed set of
  standardized observations (not on their own trajectories) so it isolates
  differences in the policy itself.
- **Behavioral repeatability**: a one-way random-effects ICC computed from
  first principles (between-agent variance vs. within-agent variance across
  repeated standardized trials) on 9 behavioral features (resources
  collected, exploration coverage, action entropy, movement-direction
  balance, time spent near other agents).
- **Temporal stability**: Pearson correlation of an agent's behavior-profile
  vector between consecutive checkpoints, averaged across agents.
- **Cross-context consistency**: Spearman rank correlation of agents'
  behavior profiles between two different standardized test environments
  (context B uses a smaller grid and higher resource density than context A).

## Known limitations (stated plainly)

- Small hand-rolled networks (default hidden size 32) trained with
  vanilla REINFORCE — expect noisier learning curves than a tuned PPO run,
  especially at low episode counts. Increase `num_episodes` and
  `hidden_dim` for cleaner trends.
- No entropy bonus in the policy loss; exploration relies solely on the
  stochastic softmax policy. Reduce `learning_rate` if a run seems to
  collapse to a single action too quickly.
- Resource contests are resolved by agent index, not truly simultaneous
  — a minor, documented simplification of the collision rule.
