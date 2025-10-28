#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SARSA sweep runner:
- Sweep eta/gamma/epsilon and run your maze SARSA
- Save: steps-vs-episode line plots, final-Q heatmaps, and a summary CSV

Usage example:
  python sarsa_sweep.py --eta 0.1 0.2 0.4 0.6 0.8 --gamma 0.9 0.6 0.4 0.2 \
                        --epsilon 0.5 0.25 0.125 --episodes 20 --outdir outputs
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------- Environment & SARSA (元コードをそのまま流用+関数化) --------------


def simple_convert_into_pi_from_theta(theta: np.ndarray) -> np.ndarray:
    m, n = theta.shape
    pi = np.zeros((m, n))
    for i in range(m):
        denom = np.nansum(theta[i, :])
        if denom == 0 or np.isnan(denom):
            pi[i, :] = 0.0
        else:
            pi[i, :] = theta[i, :] / denom
    return np.nan_to_num(pi)


def get_action(s: int, Q: np.ndarray, epsilon: float, pi_0: np.ndarray) -> int:
    direction = ["up", "right", "down", "left"]
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]

    return {"up": 0, "right": 1, "down": 2, "left": 3}[next_direction]


def get_s_next(s: int, a: int) -> int:
    # 3x3 風グリッド：移動で +/−1, +/−3
    if a == 0:  # up
        return s - 3
    elif a == 1:  # right
        return s + 1
    elif a == 2:  # down
        return s + 3
    elif a == 3:  # left
        return s - 1
    raise ValueError("invalid action")


def sarsa_update(s, a, r, s_next, a_next, Q, eta, gamma):
    if s_next == 8:  # goal
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])
    return Q


def run_one_episode(Q, epsilon, eta, gamma, pi_0, goal_state=8):
    s = 0
    a = a_next = get_action(s, Q, epsilon, pi_0)
    steps = 0
    while True:
        a = a_next
        s_next = get_s_next(s, a)

        if s_next == goal_state:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi_0)

        Q = sarsa_update(s, a, r, s_next, a_next, Q, eta, gamma)

        steps += 1
        if s_next == goal_state:
            break
        s = s_next
    return steps, Q


def run_sarsa(eta, gamma, epsilon0, episodes, seed=0):
    """
    Returns:
      steps_list: list[int] length=episodes
      Q_final: (8,4) array
    """
    rng = np.random.RandomState(seed)
    # θ と π（壁は nan）
    theta_0 = np.array(
        [
            [np.nan, 1, 1, np.nan],  # s0
            [np.nan, 1, np.nan, 1],  # s1
            [np.nan, np.nan, 1, 1],  # s2
            [1, 1, 1, np.nan],  # s3
            [np.nan, np.nan, 1, 1],  # s4
            [1, np.nan, np.nan, np.nan],  # s5
            [1, np.nan, np.nan, np.nan],  # s6
            [1, 1, np.nan, np.nan],  # s7
        ],
        dtype=float,
    )
    pi_0 = simple_convert_into_pi_from_theta(theta_0)
    a, b = theta_0.shape
    Q = rng.rand(a, b) * theta_0  # 壁は nan のまま残る

    epsilon = epsilon0
    steps_list = []
    for _ in range(episodes):
        # 元コードと同じく各エピソードで epsilon を 1/2 に減衰
        epsilon *= 0.5
        steps, Q = run_one_episode(Q, epsilon, eta, gamma, pi_0)
        steps_list.append(steps)
    return steps_list, Q


# -------------- Plot helpers --------------

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
    # Q は (8,4)。壁の nan は最小値近辺に落ちないように mask する
    Q_masked = np.ma.masked_invalid(Q)
    plt.figure(figsize=(7, 3.8))
    im = plt.imshow(Q_masked, aspect="auto")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.9, 0.8, 0.6, 0.4, 0.2])
    ap.add_argument("--epsilon", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # まとめ CSV（長い形式）
    rows = []

    for eta, gamma, eps in itertools.product(args.eta, args.gamma, args.epsilon):
        steps, Q = run_sarsa(eta=eta, gamma=gamma, epsilon0=eps, episodes=args.episodes, seed=args.seed)

        tag = f"eta{eta}_gamma{gamma}_eps{eps}"
        # ファイル名に使えるように
        safe_tag = tag.replace(".", "p")

        # 1) Steps plot
        steps_png = os.path.join(args.outdir, f"steps_{safe_tag}.png")
        save_steps_plot(args.episodes, steps, title=f"Steps to Goal ({tag})", out_png=steps_png)

        # 2) Final Q heatmap
        q_png = os.path.join(args.outdir, f"qheat_{safe_tag}.png")
        save_q_heatmap(Q, title=f"Final Q Heatmap ({tag})", out_png=q_png)

        # 3) 行データ
        for ep, st in enumerate(steps, start=1):
            rows.append({"eta": eta, "gamma": gamma, "epsilon0": eps, "episode": ep, "steps": st})

        # 4) 最終Qも保存（解析用）
        q_df = pd.DataFrame(Q, columns=ACTIONS)
        q_df.insert(0, "state", [f"s{i}" for i in range(8)])
        q_csv = os.path.join(args.outdir, f"finalQ_{safe_tag}.csv")
        q_df.to_csv(q_csv, index=False)

    # まとめ CSV 出力
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.outdir, "summary_steps.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved outputs to: {os.path.abspath(args.outdir)}")
    print(f"- Step curves: steps_*.png")
    print(f"- Final Q heatmaps: qheat_*.png")
    print(f"- Final Q CSVs: finalQ_*.csv")
    print(f"- Summary steps CSV: {summary_path}")


if __name__ == "__main__":
    main()
