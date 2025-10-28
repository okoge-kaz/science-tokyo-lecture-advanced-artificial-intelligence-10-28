#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# === 入力ファイルの場所 ===
SUMMARY_CSV = "outputs/summary_steps.csv"  # sarsa_sweep.py が吐いたCSV
FINALQ_GLOB = "outputs/finalQ_*.csv"  # 最終Q値CSV

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)


# ========== ユーティリティ ==========
def most_common_combo(series_of_tuples):
    """最頻出の (param1, param2) を返す。"""
    c = Counter(series_of_tuples)
    return c.most_common(1)[0][0] if c else None


def plot_steps_by_param(df, vary, fixed_pair=None, out_png=""):
    """
    df: summary_steps.csv を読んだ DataFrame（列: eta, gamma, epsilon0, episode, steps）
    vary: 'eta' or 'gamma' or 'epsilon0'
    fixed_pair: 変えない2つの(名前,値)の辞書。Noneなら最頻出の組を自動選択
    """
    other = [c for c in ["eta", "gamma", "epsilon0"] if c != vary]
    if fixed_pair is None:
        # 最も多く出現する（other[0], other[1]）の組を選ぶ
        pair = most_common_combo(list(zip(df[other[0]], df[other[1]])))
        if pair is None:
            raise RuntimeError("データが空です")
        fixed_pair = {other[0]: pair[0], other[1]: pair[1]}

    sel = df[(df[other[0]] == fixed_pair[other[0]]) & (df[other[1]] == fixed_pair[other[1]])]
    if sel.empty:
        raise RuntimeError(f"条件に一致するデータがありません: fixed={fixed_pair}")

    plt.figure(figsize=(7.5, 5))
    for val, g in sel.groupby(vary):
        # 複数試行があっても episode ごとに平均化
        mean_steps = g.groupby("episode")["steps"].mean().sort_index()
        plt.plot(mean_steps.index, mean_steps.values, marker="o", label=f"{vary}={val}")

    ttl = f"Steps to Goal (vary {vary}; fixed {other[0]}={fixed_pair[other[0]]}, {other[1]}={fixed_pair[other[1]]})"
    plt.title(ttl)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=220)
        plt.close()
    else:
        plt.show()


# ファイル名から eta, gamma, epsilon を復元
TAG_RE = re.compile(r"finalQ_eta(?P<eta>[0-9p]+)_gamma(?P<gamma>[0-9p]+)_eps(?P<eps>[0-9p]+)\.csv$")


def parse_vals_from_filename(path):
    m = TAG_RE.search(os.path.basename(path))
    if not m:
        return None

    def p_to_float(s):
        return float(s.replace("p", "."))

    return p_to_float(m["eta"]), p_to_float(m["gamma"]), p_to_float(m["eps"])


def read_all_Q():
    """finalQ_* を全て読み、[{eta,gamma,epsilon0,Q}] のリストを返す。"""
    items = []
    for fp in glob.glob(FINALQ_GLOB):
        parsed = parse_vals_from_filename(fp)
        if not parsed:
            continue
        eta, gamma, eps = parsed
        df = pd.read_csv(fp)
        Q = df[["up", "right", "down", "left"]].to_numpy(dtype=float)
        items.append({"eta": eta, "gamma": gamma, "epsilon0": eps, "Q": Q, "path": fp})
    return items


def make_q_grid(items, vary, fixed_pair=None, out_png=""):
    """
    items: read_all_Q() の結果
    vary: 'eta' or 'gamma' or 'epsilon0'
    fixed_pair: 他2パラメータを固定する辞書。None なら最頻出を自動選択
    """
    other = [c for c in ["eta", "gamma", "epsilon0"] if c != vary]
    if fixed_pair is None:
        combos = [(d[other[0]], d[other[1]]) for d in items]
        pair = most_common_combo(combos)
        if pair is None:
            raise RuntimeError("Qデータが空です")
        fixed_pair = {other[0]: pair[0], other[1]: pair[1]}

    # 抽出 & 並べ替え
    sel = [d for d in items if (d[other[0]] == fixed_pair[other[0]] and d[other[1]] == fixed_pair[other[1]])]
    if not sel:
        raise RuntimeError(f"条件に一致するQデータがありません: fixed={fixed_pair}")
    sel = sorted(sel, key=lambda x: x[vary])

    n = len(sel)
    cols = min(n, 5)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    vmax = np.nanmax([np.nanmax(d["Q"]) for d in sel])
    vmin = np.nanmin([np.nanmin(d["Q"]) for d in sel])

    for i, d in enumerate(sel):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        Qm = np.ma.masked_invalid(d["Q"])
        im = ax.imshow(Qm, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{vary}={d[vary]}")
        ax.set_xlabel("Action")
        ax.set_ylabel("State")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["up", "right", "down", "left"])
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"s{i}" for i in range(8)])

    # 余白サブプロットを消す
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"Final Q Heatmaps (vary {vary}; fixed {other[0]}={fixed_pair[other[0]]}, {other[1]}={fixed_pair[other[1]]})",
        y=0.995,
    )

    # === カラーバーを右端に固定配置 ===
    plt.subplots_adjust(right=0.88, wspace=0.3, hspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Q value")

    if out_png:
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ========== 実行部 ==========
if __name__ == "__main__":
    # 1) steps プロット（eta / gamma / epsilon でそれぞれ）
    df = pd.read_csv(SUMMARY_CSV)

    plot_steps_by_param(df, "eta", fixed_pair=None, out_png=os.path.join(OUTDIR, "steps_vary_eta.png"))
    plot_steps_by_param(df, "gamma", fixed_pair=None, out_png=os.path.join(OUTDIR, "steps_vary_gamma.png"))
    plot_steps_by_param(df, "epsilon0", fixed_pair=None, out_png=os.path.join(OUTDIR, "steps_vary_epsilon.png"))

    # 2) Qヒートマップを1図に集約（varyごとに1図）
    items = read_all_Q()
    make_q_grid(items, "eta", fixed_pair=None, out_png=os.path.join(OUTDIR, "Qheat_vary_eta.png"))
    make_q_grid(items, "gamma", fixed_pair=None, out_png=os.path.join(OUTDIR, "Qheat_vary_gamma.png"))
    make_q_grid(items, "epsilon0", fixed_pair=None, out_png=os.path.join(OUTDIR, "Qheat_vary_epsilon.png"))

    print(f"✔ Saved plots under: {OUTDIR}")
    print("  - steps_vary_eta.png / steps_vary_gamma.png / steps_vary_epsilon.png")
    print("  - Qheat_vary_eta.png / Qheat_vary_gamma.png / Qheat_vary_epsilon.png")
