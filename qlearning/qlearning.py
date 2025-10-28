#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q-learning sweep runner:
Sweep eta/gamma/epsilon combinations for maze Q-learning,
save step curves, Q heatmaps, and summary CSV.

Usage example:
  python qlearning_sweep.py --eta 0.1 0.2 0.4 --gamma 0.9 0.6 0.3 \
                            --epsilon 0.5 0.25 --episodes 20 --outdir qout
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========== Q-learning core ==========
def simple_convert_into_pi_from_theta(theta):
    m, n = theta.shape
    pi = np.zeros((m, n))
    for i in range(m):
        denom = np.nansum(theta[i, :])
        pi[i, :] = theta[i, :] / denom if denom != 0 else 0
    return np.nan_to_num(pi)


def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]
    return {"up": 0, "right": 1, "down": 2, "left": 3}[next_direction]


def get_s_next(s, a):
    if a == 0:
        return s - 3
    if a == 1:
        return s + 1
    if a == 2:
        return s + 3
    if a == 3:
        return s - 1


def q_learning_update(s, a, r, s_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q


def run_one_episode(Q, epsilon, eta, gamma, pi_0):
    s = 0
    steps = 0
    while True:
        a = get_action(s, Q, epsilon, pi_0)
        s_next = get_s_next(s, a)
        if s_next == 8:
            r = 1
            Q = q_learning_update(s, a, r, s_next, Q, eta, gamma)
            steps += 1
            break
        else:
            r = 0
            Q = q_learning_update(s, a, r, s_next, Q, eta, gamma)
            s = s_next
            steps += 1
    return steps, Q


def run_qlearning(eta, gamma, epsilon0, episodes, seed=0):
    np.random.seed(seed)
    theta_0 = np.array(
        [
            [np.nan, 1, 1, np.nan],
            [np.nan, 1, np.nan, 1],
            [np.nan, np.nan, 1, 1],
            [1, 1, 1, np.nan],
            [np.nan, np.nan, 1, 1],
            [1, np.nan, np.nan, np.nan],
            [1, np.nan, np.nan, np.nan],
            [1, 1, np.nan, np.nan],
        ]
    )
    pi_0 = simple_convert_into_pi_from_theta(theta_0)
    a, b = theta_0.shape
    Q = np.random.rand(a, b) * theta_0

    epsilon = epsilon0
    steps_list = []
    for _ in range(episodes):
        epsilon *= 0.5
        steps, Q = run_one_episode(Q, epsilon, eta, gamma, pi_0)
        steps_list.append(steps)
    return steps_list, Q


# ========== Plot helpers ==========
ACTIONS = ["up", "right", "down", "left"]


def save_steps_plot(episodes, steps, title, out_png):
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, episodes + 1), steps, marker="o")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_q_heatmap(Q, title, out_png):
    Qm = np.ma.masked_invalid(Q)
    plt.figure(figsize=(7, 3.5))
    im = plt.imshow(Qm, aspect="auto")
    plt.title(title)
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.xticks(range(4), ACTIONS)
    plt.yticks(range(8), [f"s{i}" for i in range(8)])
    cbar = plt.colorbar(im)
    cbar.set_label("Q value")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ========== Main sweep runner ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.9, 0.6, 0.4, 0.2])
    ap.add_argument("--epsilon", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="qout")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []

    for eta, gamma, eps in itertools.product(args.eta, args.gamma, args.epsilon):
        steps, Q = run_qlearning(eta=eta, gamma=gamma, epsilon0=eps, episodes=args.episodes, seed=args.seed)
        tag = f"eta{eta}_gamma{gamma}_eps{eps}"
        safe_tag = tag.replace(".", "p")
        # 1) steps plot
        steps_png = os.path.join(args.outdir, f"steps_{safe_tag}.png")
        save_steps_plot(args.episodes, steps, f"Q-learning Steps ({tag})", steps_png)
        # 2) Q heatmap
        q_png = os.path.join(args.outdir, f"qheat_{safe_tag}.png")
        save_q_heatmap(Q, f"Final Q Heatmap ({tag})", q_png)
        # 3) CSV outputs
        for ep, st in enumerate(steps, start=1):
            rows.append({"eta": eta, "gamma": gamma, "epsilon0": eps, "episode": ep, "steps": st})
        q_df = pd.DataFrame(Q, columns=ACTIONS)
        q_df.insert(0, "state", [f"s{i}" for i in range(8)])
        q_df.to_csv(os.path.join(args.outdir, f"finalQ_{safe_tag}.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary_steps.csv"), index=False)
    print(f"✔ Results saved in: {args.outdir}")


if __name__ == "__main__":
    main()
