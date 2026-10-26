"""Phase 3 (stretch) on REAL data: same contextual-bandit zone-adjustment
simulation as the synthetic study, now driven by real PmatData pressure
frames as the context stream instead of synthetic ones. Only the context
source changes; the hidden comfort function, policies, and regret
definition are identical to scripts/06_bandit.py."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.real_data import load_subjects, TRAIN_SUBJECTS, H

ZONES = ["head", "shoulders", "lower_back_hips", "thighs", "calves"]
K = len(ZONES)
D = K + 1
T = 3000
REWARD_NOISE_STD = 0.05


def zone_context(pressure_map: np.ndarray) -> np.ndarray:
    bounds = np.linspace(0, H, K + 1).astype(int)
    means = np.array([pressure_map[bounds[i]:bounds[i + 1]].mean() for i in range(K)])
    return np.concatenate([means, [1.0]])


def make_theta(rng):
    theta = np.zeros((K, D))
    sensitivities = rng.uniform(0.5, 2.0, size=K)
    for z in range(K):
        theta[z, z] = sensitivities[z]
        theta[z, -1] = rng.normal(0, 0.05)
    return theta


def true_rewards(theta, x):
    return theta @ x


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

    # real pressure frames (train subjects, all postures pooled) as the context stream
    data = load_subjects(TRAIN_SUBJECTS)
    frame_rng = np.random.default_rng(99)
    n_frames = data["clean"].shape[0]
    sel = frame_rng.choice(n_frames, size=T, replace=(T > n_frames))
    contexts = np.stack([zone_context(data["clean"][i]) for i in sel])
    print(f"using {n_frames} real frames as context pool, sampled {T} rounds")

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
    with open("reports/results/real_bandit_results.json", "w") as f:
        json.dump(dict(zones=ZONES, T=T, theta_sensitivities=theta[np.arange(K), np.arange(K)].tolist(),
                        results=results, context_source="real PmatData train-subject frames"), f, indent=2)

    plt.figure(figsize=(7, 5))
    pretty = {"random": "Random", "eps_greedy": "Epsilon-greedy (eps=0.1)", "linucb": "LinUCB (alpha=1.0)", "oracle": "Oracle (upper bound)"}
    for name in policies:
        plt.plot(curves[name], label=pretty[name])
    plt.xlabel("round t"); plt.ylabel("cumulative regret")
    plt.title("Contextual bandit on real pressure data")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/12_real_bandit_regret.png", dpi=130)
    print("saved reports/figures/12_real_bandit_regret.png")


if __name__ == "__main__":
    main()
