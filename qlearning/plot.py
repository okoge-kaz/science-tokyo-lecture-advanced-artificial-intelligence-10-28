import os
import glob
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

SUMMARY_CSV = "qout/summary_steps.csv"
FINALQ_GLOB = "qout/finalQ_*.csv"
OUTDIR = "qplots"
os.makedirs(OUTDIR, exist_ok=True)


def most_common_combo(series_of_tuples):
    c = Counter(series_of_tuples)
    return c.most_common(1)[0][0] if c else None


def plot_steps_by_param(df, vary, fixed_pair=None, out_png=""):
    other = [c for c in ["eta", "gamma", "epsilon0"] if c != vary]
    if fixed_pair is None:
        pair = most_common_combo(list(zip(df[other[0]], df[other[1]])))
        fixed_pair = {other[0]: pair[0], other[1]: pair[1]}
    sel = df[(df[other[0]] == fixed_pair[other[0]]) & (df[other[1]] == fixed_pair[other[1]])]

    plt.figure(figsize=(8, 5))
    for val, g in sel.groupby(vary):
        mean_steps = g.groupby("episode")["steps"].mean().sort_index()
        plt.plot(mean_steps.index, mean_steps.values, marker="o", label=f"{vary}={val}")

    plt.title(
        f"Steps to Goal (vary {vary}; fixed {other[0]}={fixed_pair[other[0]]}, {other[1]}={fixed_pair[other[1]]})"
    )
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, linestyle="--", alpha=0.6)

    # --- 凡例を右側に配置 ---
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.subplots_adjust(right=0.8)  # 右に凡例スペースを確保

    if out_png:
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


TAG_RE = re.compile(r"finalQ_eta(?P<eta>[0-9p]+)_gamma(?P<gamma>[0-9p]+)_eps(?P<eps>[0-9p]+)\.csv$")


def parse_vals_from_filename(path):
    m = TAG_RE.search(os.path.basename(path))
    if not m:
        return None

    def p_to_float(s):
        return float(s.replace("p", "."))

    return p_to_float(m["eta"]), p_to_float(m["gamma"]), p_to_float(m["eps"])


def read_all_Q():
    items = []
    for fp in glob.glob(FINALQ_GLOB):
        parsed = parse_vals_from_filename(fp)
        if not parsed:
            continue
        eta, gamma, eps = parsed
        df = pd.read_csv(fp)
        Q = df[["up", "right", "down", "left"]].to_numpy(dtype=float)
        items.append({"eta": eta, "gamma": gamma, "epsilon0": eps, "Q": Q})
    return items


def make_q_grid(items, vary, fixed_pair=None, out_png=""):
    """
    items: read_all_Q() の結果
    vary: 'eta' or 'gamma' or 'epsilon0'
    fixed_pair: 他2パラメータを固定する辞書。None なら最頻出を自動選択
    """
    other = [c for c in ["eta", "gamma", "epsilon0"] if c != vary]
    if fixed_pair is None:
        pair = most_common_combo([(d[other[0]], d[other[1]]) for d in items])
        fixed_pair = {other[0]: pair[0], other[1]: pair[1]}

    sel = [d for d in items if (d[other[0]] == fixed_pair[other[0]] and d[other[1]] == fixed_pair[other[1]])]
    sel = sorted(sel, key=lambda x: x[vary])
    n = len(sel)
    cols = min(n, 5)
    rows = math.ceil(n / cols)

    # --- カラーバー用に右側余白を確保 ---
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols + 1.2, 3.2 * rows),  # +1.2 は右側の余白
        squeeze=False,
        gridspec_kw={"wspace": 0.25, "hspace": 0.3, "width_ratios": [1] * cols},
    )

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

    # 空の余白を消す
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    # --- カラーバーを最右側に配置 ---
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("Q value", rotation=270, labelpad=15)

    fig.suptitle(
        f"Final Q Heatmaps (vary {vary}; fixed {other[0]}={fixed_pair[other[0]]}, {other[1]}={fixed_pair[other[1]]})",
        y=0.99,
    )

    plt.subplots_adjust(right=0.9, left=0.08, bottom=0.1, top=0.9)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(SUMMARY_CSV)
    plot_steps_by_param(df, "eta", out_png=os.path.join(OUTDIR, "steps_vary_eta.png"))
    plot_steps_by_param(df, "gamma", out_png=os.path.join(OUTDIR, "steps_vary_gamma.png"))
    plot_steps_by_param(df, "epsilon0", out_png=os.path.join(OUTDIR, "steps_vary_epsilon.png"))
    items = read_all_Q()
    make_q_grid(items, "eta", out_png=os.path.join(OUTDIR, "Qheat_vary_eta.png"))
    make_q_grid(items, "gamma", out_png=os.path.join(OUTDIR, "Qheat_vary_gamma.png"))
    make_q_grid(items, "epsilon0", out_png=os.path.join(OUTDIR, "Qheat_vary_epsilon.png"))
    print(f"✔ Saved Q-learning summary plots under {OUTDIR}")
