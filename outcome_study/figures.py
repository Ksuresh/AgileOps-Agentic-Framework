"""Paper figures for the confirmatory outcome studies (14-16), recomputed from committed data.
Fig. A  pooled O1 accuracy with Wilson 95% intervals, by decider family (dot plot).
Fig. B  accuracy by case type: where each decider fails (heatmap, share correct).
Fig. C  paired comparison with AAF: cases only AAF got right vs. cases only the comparator got right.
Usage: python outcome_study/figures.py [outdir]   (default paper_figures/)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import wilson  # noqa: E402
from outcome_study.analyze_gpt import load  # noqa: E402

STUDIES = ("outcome_study14", "outcome_study15", "outcome_study16")
import os
PREVIEW = bool(os.environ.get("PREVIEW"))
FAULT = "RPSKDX"
GROUP_NAMES = {"R": "Release regression", "P": "Fault before release", "S": "Security vs. cost",
               "K": "Security, others healthy", "X": "Compound", "D": "Cost only"}
# decider -> (label, family); siloed rule experts use v1.4 rules where the study has them (Studies 15-16) and the
# post hoc v1.4 recomputation is not mixed in for Study 14 (v1.0 there, as pre-specified)
SHOW = [
    ("AAF-14-GPT", "AAF, GPT-5.6 Luna investigator", "AAF"),
    ("AAF-14", "AAF, Qwen3-8B investigator", "AAF"),
    ("B1-14", "AAF rules alone (no investigation)", "AAF"),
    ("B6-GPT", "GPT-5.6 Luna decides, with AAF's rules", "LLM decides"),
    ("B5-GPT", "GPT-5.6 Luna decides", "LLM decides"),
    ("B6", "Qwen3-8B decides, with AAF's rules", "LLM decides"),
    ("B5", "Qwen3-8B decides", "LLM decides"),
    ("AlwaysRollback", "Always roll back", "Fixed policy"),
    ("Dominant", "Act on most severe domain", "Fixed policy"),
    ("BlockAnyCVE", "Block on any vulnerability", "Fixed policy"),
    ("D-Sec", "Security expert (rules)", "Single domain"),
    ("D-DevOps", "DevOps expert (rules)", "Single domain"),
    ("D-FinOps", "FinOps expert (rules)", "Single domain"),
    ("D-SRE", "SRE expert (rules)", "Single domain"),
    ("L-Sec", "Security expert (LLM)", "Single domain"),
    ("L-SRE", "SRE expert (LLM)", "Single domain"),
]
FAMILY_STYLE = {  # validated categorical slots 1-4 (reference palette) + distinct markers for grayscale print
    "AAF": ("#2a78d6", "D"), "LLM decides": ("#eb6834", "o"), "Fixed policy": ("#1baf7a", "s"), "Single domain": ("#6250d6", "^")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def collect():
    """Per decider: list of (study, case, group, correct) over fault cases; AAF = AAF-14 (Qwen investigator)."""
    rows = {}
    for st in STUDIES:
        A, table, dec, ok, maj, _ = load(st)
        alias = {}
        for key, _, _ in SHOW:
            src = alias.get(key, key)
            if key.startswith("D-") and (src + "-14") in dec:
                src = src + "-14"
            for c in sorted(table):
                g = c.split("-")[1][0]
                s2 = src
                if g in FAULT and dec[s2].get(c):
                    rows.setdefault(key, []).append((st, c, g, ok(s2, c)))
    if PREVIEW:  # drop deciders whose runs are incomplete
        full = max(len(v) for v in rows.values())
        rows = {k: v for k, v in rows.items() if len(v) == full}
    return rows


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_accuracy(rows, out: Path):
    items = [(k, lab, fam) for k, lab, fam in SHOW if k in rows]
    fig, ax = plt.subplots(figsize=(3.45, 0.24 * len(items) + 0.7))
    for i, (k, lab, fam) in enumerate(items):
        n, x = len(rows[k]), sum(r[3] for r in rows[k])
        lo, hi = wilson(x, n)
        col, mk = FAMILY_STYLE[fam]
        y = len(items) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(100 * x / n, y, marker=mk, ms=6, color=col, mec="white", mew=1.0, zorder=3)
        ax.text(101.5, y, f"{x}/{n}", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab for _, lab, _ in reversed(items)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Outcome-correct actions on fault cases, % (Wilson 95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=5, label=f) for f, (c, m) in FAMILY_STYLE.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=4, frameon=False, fontsize=6.5,
              handletextpad=0.2, columnspacing=0.8)
    fig.savefig(out / "fig_outcome_accuracy.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_outcome_accuracy.png", dpi=200, bbox_inches="tight")


def fig_groups(rows, out: Path):
    keys = [k for k in ("AAF-14-GPT", "B6-GPT", "B5-GPT", "AlwaysRollback", "BlockAnyCVE", "D-Sec", "D-DevOps", "D-SRE", "D-FinOps") if k in rows]
    labels = {k: lab for k, lab, _ in SHOW}
    short = {"AAF-14-GPT": "AAF", "B6-GPT": "GPT+rules", "B5-GPT": "GPT", "AlwaysRollback": "Rollback", "BlockAnyCVE": "Block CVE",
             "D-Sec": "Security", "D-DevOps": "DevOps", "D-SRE": "SRE", "D-FinOps": "FinOps"}
    groups = [g for g in "RPSKXD"]
    import numpy as np
    M = np.zeros((len(groups), len(keys)))
    N = np.zeros((len(groups), len(keys)), dtype=int)
    for j, k in enumerate(keys):
        for i, g in enumerate(groups):
            rr = [r for r in rows[k] if r[2] == g]
            N[i, j] = len(rr)
            M[i, j] = (sum(r[3] for r in rr) / len(rr)) if rr else np.nan
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#cfe1f7", "#7fb0ea", "#2a78d6", "#123f77"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(groups)):
        for j in range(len(keys)):
            v = M[i, j]
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v >= 0.6 else INK)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([short[k] for k in keys], rotation=40, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{GROUP_NAMES[g]} ({N[i, 0]})" for i, g in enumerate(groups)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(keys))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(groups))], minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    fig.savefig(out / "fig_outcome_groups.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_outcome_groups.png", dpi=200, bbox_inches="tight")


def fig_paired(rows, out: Path):
    comps = [k for k, _, _ in SHOW if k != "AAF" and k in rows]
    labels = {k: lab for k, lab, _ in SHOW}
    a = {(r[0], r[1]): r[3] for r in rows["AAF-14-GPT"]}
    data = []
    for k in comps:
        b = {(r[0], r[1]): r[3] for r in rows[k]}
        common = [x for x in b if x in a]
        data.append((k, sum(a[x] and not b[x] for x in common), sum(b[x] and not a[x] for x in common)))
    fig, ax = plt.subplots(figsize=(3.45, 0.22 * len(data) + 0.6))
    for i, (k, w, l) in enumerate(data):
        y = len(data) - 1 - i
        ax.barh(y, w, color="#2a78d6", height=0.62)
        ax.barh(y, -l, color="#eb6834", height=0.62)
        ax.text(w + 0.8, y, str(w), va="center", fontsize=6.5, color=INK2)
        if l:
            ax.text(-l - 0.8, y, str(l), va="center", ha="right", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([labels[k] for k, _, _ in reversed(data)])
    ax.set_xlabel("Fault cases where only one of the pair chose correctly")
    lim = max(max(w for _, w, _ in data), 5) + 8
    ax.set_xlim(-12, lim)
    ax.text(lim * 0.55, len(data) - 0.3, "only AAF correct", color="#2a78d6", fontsize=6.5, ha="center")
    ax.text(-6, len(data) - 0.3, "only other", color="#c4521f", fontsize=6.5, ha="center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "fig_outcome_paired.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_outcome_paired.png", dpi=200, bbox_inches="tight")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    rows = collect()
    fig_accuracy(rows, out)
    fig_groups(rows, out)
    fig_paired(rows, out)
    for k, v in rows.items():
        print(k, sum(r[3] for r in v), "/", len(v))


if __name__ == "__main__":
    main()
