"""Phase 3 (stretch): contextual bandit for "which mattress zone to adjust".

Setup
-----
At each round the system observes the current per-zone pressure context
(K=5 zones along the body: head, shoulders, lower-back/hips, thighs,
calves) and must choose ONE zone to relieve. The simulated user has a
*hidden* per-zone comfort-sensitivity vector theta (unknown to the agent):
  reward(zone z | context x) = theta_z * x[z] + noise
i.e. the payoff of relieving a zone is proportional to how much pressure is
currently there, scaled by how much that particular user cares about that
zone -- a disjoint linear contextual bandit, the standard LinUCB setting.

Policies compared
------------------
  random      : picks a zone uniformly at random (no learning)
  eps_greedy  : per-arm ridge regression estimate, epsilon=0.1 random exploration
  linucb      : disjoint LinUCB (Li et al., 2010) with confidence-bound exploration
  oracle      : knows theta exactly, always picks the true best zone (regret=0 by construction)

We report cumulative regret (oracle reward - chosen reward, under the true
noise-free reward model) over time for each policy.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import sample_batch, H

ZONES = ["head", "shoulders", "lower_back_hips", "thighs", "calves"]
K = len(ZONES)
D = K + 1  # context = per-zone mean pressure + bias term
T = 3000
REWARD_NOISE_STD = 0.05


def zone_context(pressure_map: np.ndarray) -> np.ndarray:
    """H x W pressure map -> (K,) mean pressure per zone band, + bias term."""
    bounds = np.linspace(0, H, K + 1).astype(int)
    means = np.array([pressure_map[bounds[i]:bounds[i + 1]].mean() for i in range(K)])
    return np.concatenate([means, [1.0]])  # append bias dim


def make_theta(rng):
    """Hidden per-zone comfort sensitivities for one simulated user.
    theta_z is nonzero only on dimension z (disjoint linear bandit) plus a
    small shared bias-sensitivity, so reward depends mainly on *that* zone's
    pressure, scaled by how much this particular user is bothered by it."""
    theta = np.zeros((K, D))
    sensitivities = rng.uniform(0.5, 2.0, size=K)
    for z in range(K):
        theta[z, z] = sensitivities[z]
        theta[z, -1] = rng.normal(0, 0.05)  # small per-arm bias
    return theta


def true_rewards(theta, x):
    return theta @ x  # (K,)


class EpsGreedy:
    def __init__(self, k, d, epsilon=0.1, ridge=1.0):
        self.k, self.d, self.epsilon = k, d, epsilon
        self.A = [ridge * np.eye(d) for _ in range(k)]
        self.b = [np.zeros(d) for _ in range(k)]

    def select(self, x, rng):
        if rng.random() < self.epsilon:
            return rng.integers(0, self.k)
        est = [np.linalg.solve(self.A[a], self.b[a]) @ x for a in range(self.k)]
        return int(np.argmax(est))

    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


class LinUCB:
    def __init__(self, k, d, alpha=1.0, ridge=1.0):
        self.k, self.d, self.alpha = k, d, alpha
        self.A = [ridge * np.eye(d) for _ in range(k)]
        self.b = [np.zeros(d) for _ in range(k)]

    def select(self, x, rng):
        scores = []
        for a in range(self.k):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            mean = theta_hat @ x
            ucb = self.alpha * np.sqrt(x @ A_inv @ x)
            scores.append(mean + ucb)
        return int(np.argmax(scores))

    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


def run_policy(policy_name, theta, contexts, rng):
    cum_regret = np.zeros(T)
    rewards_log = np.zeros(T)
    policy = None
    if policy_name == "eps_greedy":
        policy = EpsGreedy(K, D)
    elif policy_name == "linucb":
        policy = LinUCB(K, D, alpha=1.0)

    running_regret = 0.0
    for t in range(T):
        x = contexts[t]
        true_r = true_rewards(theta, x)
        best_r = true_r.max()

        if policy_name == "random":
            a = rng.integers(0, K)
        elif policy_name == "oracle":
            a = int(np.argmax(true_r))
        else:
            a = policy.select(x, rng)

        observed_r = true_r[a] + rng.normal(0, REWARD_NOISE_STD)
        if policy is not None:
            policy.update(a, x, observed_r)

        running_regret += best_r - true_r[a]
        cum_regret[t] = running_regret
        rewards_log[t] = true_r[a]

    return cum_regret, rewards_log


def main():
    rng = np.random.default_rng(42)
    theta = make_theta(rng)

    # one fixed simulated user's pressure-context stream for the whole run
    data_rng_seed = 99
    batch = sample_batch(T, seed=data_rng_seed)
    contexts = np.stack([zone_context(batch["noisy"][i]) for i in range(T)])

    policies = ["random", "eps_greedy", "linucb", "oracle"]
    policy_seeds = {"random": 1001, "eps_greedy": 1002, "linucb": 1003, "oracle": 1004}
    results = {}
    curves = {}
    for name in policies:
        policy_rng = np.random.default_rng(policy_seeds[name])
        cum_regret, rewards_log = run_policy(name, theta, contexts, policy_rng)
        curves[name] = cum_regret
        results[name] = dict(
            final_cumulative_regret=float(cum_regret[-1]),
            mean_reward=float(rewards_log.mean()),
            mean_reward_last_500=float(rewards_log[-500:].mean()),
        )
        print(f"{name:12s} final_cum_regret={cum_regret[-1]:8.2f}  mean_reward={rewards_log.mean():.4f}  "
              f"mean_reward_last500={rewards_log[-500:].mean():.4f}")

    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/bandit_results.json", "w") as f:
        json.dump(dict(zones=ZONES, T=T, theta_sensitivities=theta[np.arange(K), np.arange(K)].tolist(),
                        results=results), f, indent=2)

    plt.figure(figsize=(7, 5))
    pretty = {"random": "Random", "eps_greedy": "Epsilon-greedy (eps=0.1)", "linucb": "LinUCB (alpha=1.0)", "oracle": "Oracle (upper bound)"}
    for name in policies:
        plt.plot(curves[name], label=pretty[name])
    plt.xlabel("round t"); plt.ylabel("cumulative regret")
    plt.title("Contextual bandit: which mattress zone to adjust")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/06_bandit_regret.png", dpi=130)
    print("saved reports/figures/06_bandit_regret.png")


if __name__ == "__main__":
    main()
