#!/usr/bin/env python3
"""Make every figure and table of the report from results/*.csv.

    python3 scripts/plots.py            # -> report/figs/*.pdf, report/tables.tex, results/summary.txt

If a configuration was measured more than once (the bench script appends),
each phase keeps its fastest time, as main.c does across repetitions.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figs")
os.makedirs(FIG, exist_ok=True)

GRAPHS = ["k49_norm_10NN", "com-Amazon", "coPapersDBLP"]
GNAME = {"k49_norm_10NN": "k49_norm_10NN", "com-Amazon": "com-Amazon", "coPapersDBLP": "coPapersDBLP"}
PHASES = ["t_P", "t_attr", "t_build", "t_com", "t_walk"]
PNAME = {"t_P": "P", "t_attr": "attraction", "t_build": "tree build",
         "t_com": "centers of mass", "t_walk": "walk", "t_total": "total"}
# categorical palette, fixed order (dataviz reference palette, light mode)
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MARK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "legend.frameon": False,
    "legend.fontsize": 7, "axes.titlesize": 8.5,
})

KEY = ["graph", "version", "threads", "L", "theta"]


def load(name):
    d = pd.read_csv(os.path.join(RES, name))
    tcols = [c for c in d.columns if c.startswith("t_")]
    agg = {c: "min" for c in tcols}
    for c in d.columns:
        if c not in agg and c not in KEY:
            agg[c] = "first"
    return d.groupby(KEY, as_index=False).agg(agg)


T = load("timings.csv")
D = load("depth.csv")
TH = load("theta.csv")
X = pd.read_csv(os.path.join(RES, "crossover.csv")).groupby(
    ["graph", "version", "threads", "m", "L"], as_index=False).min()
DIR = load("direct.csv")
THREADS = sorted(T.threads.unique())


def seq(g):
    return T[(T.graph == g) & (T.version == "seq")].iloc[0]


def row(g, v, p):
    return T[(T.graph == g) & (T.version == v) & (T.threads == p)].iloc[0]


def ptick(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(THREADS)
    ax.set_xticklabels([str(p) for p in THREADS])
    ax.minorticks_off()
    ax.axvline(8, color=INK2, lw=0.6, ls=":")


# ---- Figure 1: time of each phase against p (OpenMP) ----
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, g in zip(axs, GRAPHS):
    d = T[(T.graph == g) & (T.version == "omp")].sort_values("threads")
    for k, ph in enumerate(PHASES):
        ax.plot(d.threads, d[ph] * 1e3, color=C[k], marker=MARK[k], label=PNAME[ph])
    ax.set_yscale("log")
    ptick(ax)
    ax.set_title(GNAME[g])
    ax.set_xlabel("threads p")
axs[0].set_ylabel("time (ms), log scale")
for ax in axs:
    ax.text(8.4, ax.get_ylim()[1] * 0.55, "8 cores", color=INK2, fontsize=6)
axs[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "phases.pdf"))
plt.close(fig)

# ---- Figure 2: speedup of the total, three models ----
VERS = [("omp", "OpenMP"), ("pth", "Pthreads"), ("cilk", "OpenCilk")]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
for ax, g in zip(axs, GRAPHS):
    t1 = seq(g).t_total
    ax.plot(THREADS, THREADS, color=INK2, lw=0.8, ls="--", label="ideal")
    for k, (v, vn) in enumerate(VERS):
        d = T[(T.graph == g) & (T.version == v)].sort_values("threads")
        ax.plot(d.threads, t1 / d.t_total, color=C[k], marker=MARK[k], label=vn)
    ptick(ax)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.set_title(GNAME[g])
    ax.set_xlabel("threads p")
axs[0].set_ylabel("speedup vs sequential")
axs[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "speedup.pdf"))
plt.close(fig)

# ---- Figure 3: tree depth L (OpenMP, 16 threads) ----
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
for ax, g in zip(axs, GRAPHS):
    d = D[(D.graph == g) & (D.version == "omp") & (D.threads == 16)].sort_values("L")
    for k, ph in [(2, "t_build"), (3, "t_com"), (4, "t_walk")]:
        ax.plot(d.L, d[ph] * 1e3, color=C[k], marker=MARK[k], label=PNAME[ph])
    tree = (d.t_build + d.t_com + d.t_walk) * 1e3
    ax.plot(d.L, tree, color=INK, marker="o", lw=1.0, label="build+com+walk")
    best = d.L.iloc[tree.values.argmin()]
    ax.axvline(best, color=INK2, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(list(d.L))
    ax.set_title(GNAME[g])
    ax.text(best + 0.1, ax.get_ylim()[1] * 0.4 if ax.get_ylim()[1] > 0 else 1, f"best L = {best}",
            color=INK2, fontsize=6)
    ax.set_xlabel("depth L (grid $2^L\\times 2^L$)")
axs[0].set_ylabel("time (ms), 16 threads")
axs[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "depth.pdf"))
plt.close(fig)

# ---- Figure 4: crossover (left) and theta (right) ----
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
ax = axs[0]
for k, p in enumerate([1, 16]):
    d = X[X.threads == p].sort_values("m")
    ls = "-" if p == 1 else "--"
    ax.plot(d.m, d.t_bh * 1e3, color=C[0], ls=ls, marker=MARK[0], label=f"Barnes-Hut, p={p}")
    ax.plot(d.m, d.t_direct * 1e3, color=C[1], ls=ls, marker=MARK[1], label=f"direct $O(n^2)$, p={p}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("points m (first m of k49_norm_10NN)")
ax.set_ylabel("repulsion time (ms)")
ax.set_title("Barnes-Hut vs direct sum")
ax.legend(loc="upper left")
ax = axs[1]
for k, g in enumerate(GRAPHS):
    d = TH[(TH.graph == g)].sort_values("theta")
    ax.plot(d.err_Frep, d.t_walk * 1e3, color=C[k], marker=MARK[k], label=GNAME[g])
    for _, r in d.iterrows():
        if g == "coPapersDBLP":
            ax.annotate(f"θ={r.theta:g}", (r.err_Frep, r.t_walk * 1e3), textcoords="offset points",
                        xytext=(4, 2), fontsize=6, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("error of $F^{rep}$ vs reference")
ax.set_ylabel("walk time (ms), 16 threads")
ax.set_title("opening angle θ: accuracy vs time")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "crossover_theta.pdf"))
plt.close(fig)

# ---- Tables and numbers for the text ----
lines = []
out = []
# per-phase table: seq, best of 16 threads, speedup, per-nnz / per-point costs
out.append(r"\begin{tabular}{l r r r r r r r r r}")
out.append(r"\toprule")
out.append(r" & \multicolumn{3}{c}{k49\_norm\_10NN} & \multicolumn{3}{c}{com-Amazon} & \multicolumn{3}{c}{coPapersDBLP}\\")
out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
out.append(r"phase & $p{=}1$ & $p{=}16$ & $S$ & $p{=}1$ & $p{=}16$ & $S$ & $p{=}1$ & $p{=}16$ & $S$\\")
out.append(r"\midrule")
for ph in PHASES + ["t_total"]:
    cells = [PNAME[ph]]
    for g in GRAPHS:
        t1 = seq(g)[ph]
        t16 = row(g, "omp", 16)[ph]
        cells += [f"{t1*1e3:.1f}", f"{t16*1e3:.2f}", f"{t1/t16:.1f}"]
    if ph == "t_total":
        out.append(r"\midrule")
    out.append(" & ".join(cells) + r"\\")
out.append(r"\bottomrule")
out.append(r"\end{tabular}")
with open(os.path.join(ROOT, "report", "table_phases.tex"), "w") as f:
    f.write("\n".join(out) + "\n")

# prediction vs measurement for coPapersDBLP (OpenMP)
pred = {"t_P": (0.77, 0.15), "t_attr": (0.09, 0.015), "t_build": (0.006, 0.005),
        "t_com": (0.002, 0.0006), "t_walk": (1.16, 0.14), "t_total": (2.0, 0.31)}
out = [r"\begin{tabular}{l l r r r r}", r"\toprule",
       r"phase & model & pred.\ $p{=}1$ & meas. & pred.\ $p{=}16$ & meas.\\", r"\midrule"]
model = {"t_P": r"$\propto$ nnz", "t_attr": r"$\propto$ nnz", "t_build": r"$a n + b 4^L$",
         "t_com": r"$\propto 4^L + n$", "t_walk": r"$c\, n \log_2 n$", "t_total": ""}
for ph, (p1, p16) in pred.items():
    m1 = T[(T.graph == "coPapersDBLP") & (T.version == "omp") & (T.threads == 1)].iloc[0][ph]
    m16 = row("coPapersDBLP", "omp", 16)[ph]
    if ph == "t_total":
        out.append(r"\midrule")
    f16 = (lambda x: f"{x:.0f}") if p16 * 1e3 >= 10 else (lambda x: f"{x:.1f}")
    out.append(f"{PNAME[ph]} & {model[ph]} & {p1*1e3:.0f} & {m1*1e3:.0f} & {f16(p16*1e3)} & {f16(m16*1e3)}" + r"\\")
out += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(ROOT, "report", "table_prediction.tex"), "w") as f:
    f.write("\n".join(out) + "\n")

# numbers quoted in the text
for g in GRAPHS:
    s = seq(g)
    best = T[(T.graph == g) & (T.version != "seq")].sort_values("t_total").iloc[0]
    lines.append(f"{g}: seq total {s.t_total:.4f} s; best {best.version} p={best.threads} "
                 f"{best.t_total:.4f} s (speedup {s.t_total/best.t_total:.2f}); "
                 f"attr {s.t_attr/s.nnz*1e9:.2f} ns/nnz; P {s.t_P/s.nnz*1e9:.2f} ns/nnz; "
                 f"walk {s.t_walk/s.n*1e6:.3f} us/point")
    for v in ["omp", "pth", "cilk"]:
        r16 = row(g, v, 16); r8 = row(g, v, 8)
        lines.append(f"   {v}: speedup p=8 {s.t_total/r8.t_total:.2f}, p=16 {s.t_total/r16.t_total:.2f}; "
                     f"walk S16 {s.t_walk/r16.t_walk:.2f}, P S16 {s.t_P/r16.t_P:.2f}, attr S16 {s.t_attr/r16.t_attr:.2f}, "
                     f"build p=16 {r16.t_build*1e3:.2f} ms vs seq {s.t_build*1e3:.2f}")
for _, r in DIR.iterrows():
    tree = r.t_build + r.t_com + r.t_walk
    lines.append(f"direct k49 p={r.threads}: direct {r.t_direct:.4f} s, BH {tree:.4f} s, ratio {r.t_direct/tree:.1f}")
for p in [1, 16]:
    d = X[X.threads == p].sort_values("m")
    faster = d[d.t_direct < d.t_bh]
    lines.append(f"crossover p={p}: direct faster for m <= {faster.m.max() if len(faster) else 'none'}; "
                 + ", ".join(f"{int(m)}:{b*1e3:.3f}/{dd*1e3:.3f}" for m, b, dd in zip(d.m, d.t_bh, d.t_direct)))
with open(os.path.join(RES, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("figures in report/figs, tables in report/table_*.tex")
