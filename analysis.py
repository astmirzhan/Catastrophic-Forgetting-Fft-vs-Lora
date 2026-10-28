"""
Analysis for Assignment 4: reads every run JSON produced by forgetting.py and writes
tables, statistical tests, hypothesis verdicts, error analysis and figures.

    python analysis.py --results results_colab results_runpod --out report

Several result folders can be passed (e.g. 0.5B runs from Colab and 1.5B runs from
RunPod); they are merged.  Missing parts (e.g. no 1.5B yet) are skipped, not errors.

Outputs (in --out):
    summary.md                    human-readable digest with verdicts
    table_main.csv                mean ± SD per model / order / method / step
    per_run_steps.csv             one row per run and step (raw numbers)
    tests_per_seed.csv            McNemar + bootstrap CI per seed, Holm-corrected
    tests_across_seeds.csv        paired t-test / Wilcoxon / Cohen's dz across seeds
    correlation.csv               Spearman learning gain vs forgetting
    verdicts.csv                  H1–H3 decisions by the frozen rules
    diagnostics.csv               letter mass and letter distribution per checkpoint
    efficiency.csv                time / memory / trainable parameters
    error_sample_<method>.csv     30 forgotten items for manual labelling
    error_counts.csv              category counts over ALL forgotten items (Appendix B)
    fig1_trajectory.png ... fig6_order.png
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LETTERS = ["A", "B", "C", "D", "E"]
COLORS = {"FFT": "#2a78d6", "LoRA": "#eb6834"}
METHODS = ["FFT", "LoRA"]
N_BOOT = 2000
ALPHA = 0.05
ERROR_SEED = 42           # error analysis: smallest model, main order, seed 42, final step
N_ERROR_SAMPLE = 30

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#888888", "xtick.color": "#444444", "ytick.color": "#444444",
    "legend.frameon": False,
})


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def load_runs(folders):
    runs = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "runs", "*.json"))):
            with open(path) as f:
                r = json.load(f)
            r["_path"] = path
            runs.append(r)
    seen, unique = set(), []
    for r in runs:                       # same run in two folders -> keep the first
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            unique.append(r)
    return unique


def load_items(folders):
    for folder in folders:
        p = os.path.join(folder, "eval_items.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            return d["items"], d["sha256"]
    raise FileNotFoundError("eval_items.json not found in any results folder")


def model_label(model_id):
    return model_id.split("/")[-1]


def model_size(model_id):
    name = model_label(model_id).lower()
    for tok in name.replace("-", " ").split():
        if tok.endswith("b"):
            try:
                return float(tok[:-1])
            except ValueError:
                pass
    return 0.0


def per_run_steps(runs):
    rows = []
    for r in runs:
        c0 = np.array(r["eval"]["0"]["correct"])
        for s in r["steps"]:
            t = s["step"]
            ct = np.array(r["eval"][str(t)]["correct"])
            forgotten = int(((c0 == 1) & (ct == 0)).sum())
            gained = int(((c0 == 0) & (ct == 1)).sum())
            rows.append({
                "model": model_label(r["model"]), "size_b": model_size(r["model"]),
                "order": r["order"], "method": r["method"], "seed": r["seed"],
                "step": t, "task": s["task"], "old_acc": s["old_acc"], "ARC": s["ARC"], "MMLU": s["MMLU"],
                "forgetting_pp": s.get("forgetting_pp", 0.0),
                "learning_gain": s.get("learning_gain", np.nan),
                "loss_before": s.get("loss_before", np.nan), "loss_after": s.get("loss_after", np.nan),
                "letter_mass": s["mean_letter_mass"],
                "forgotten_items": forgotten, "gained_items": gained,
                "RFR": round(100 * forgotten / max(1, int(c0.sum())), 3),
                "train_time_s": s.get("train_time_s", np.nan), "peak_vram_gb": s.get("peak_vram_gb", np.nan),
                "trainable_params": s.get("trainable_params", np.nan),
                "total_params": r.get("total_params", np.nan), "gpu": r["environment"]["gpu"],
                "precision": r.get("precision", ""),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------
def mcnemar_exact(a, b):
    """Exact McNemar on two paired 0/1 vectors. Returns (b01, b10, p)."""
    a, b = np.asarray(a), np.asarray(b)
    n10 = int(((a == 1) & (b == 0)).sum())     # right in a, wrong in b
    n01 = int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else stats.binomtest(n10, n, 0.5).pvalue
    return n10, n01, float(p)


def bootstrap_diff(a, b, seed=0):
    """95% CI of mean(a) - mean(b) in percentage points, resampling items (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    d = (a[idx].mean(1) - b[idx].mean(1)) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tests_per_seed(runs):
    """Family H1: base vs step t (3 tests per run).  Family H2: FFT vs LoRA at step t
    (3 tests per model/order/seed).  Holm correction inside each family."""
    rows = []
    for r in runs:
        c0 = r["eval"]["0"]["correct"]
        fam = []
        for t in (1, 2, 3):
            ct = r["eval"][str(t)]["correct"]
            n10, n01, p = mcnemar_exact(c0, ct)
            lo, hi = bootstrap_diff(c0, ct, seed=r["seed"] * 10 + t)
            fam.append({"comparison": "baseline_vs_step", "model": model_label(r["model"]),
                        "order": r["order"], "method": r["method"], "seed": r["seed"], "step": t,
                        "diff_pp": round(100 * (np.mean(c0) - np.mean(ct)), 3),
                        "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                        "only_first_correct": n10, "only_second_correct": n01, "p": p})
        for row, pa in zip(fam, holm([x["p"] for x in fam])):
            row["p_holm"] = pa
        rows += fam
    by_key = {(model_label(r["model"]), r["order"], r["seed"], r["method"]): r for r in runs}
    for (m, o, s, meth), r in by_key.items():
        if meth != "FFT" or (m, o, s, "LoRA") not in by_key:
            continue
        lora = by_key[(m, o, s, "LoRA")]
        fam = []
        for t in (1, 2, 3):
            cl = lora["eval"][str(t)]["correct"]
            cf = r["eval"][str(t)]["correct"]
            n10, n01, p = mcnemar_exact(cl, cf)
            lo, hi = bootstrap_diff(cl, cf, seed=s * 100 + t)
            fam.append({"comparison": "LoRA_vs_FFT", "model": m, "order": o, "method": "LoRA-FFT",
                        "seed": s, "step": t,
                        "diff_pp": round(100 * (np.mean(cl) - np.mean(cf)), 3),
                        "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                        "only_first_correct": n10, "only_second_correct": n01, "p": p})
        for row, pa in zip(fam, holm([x["p"] for x in fam])):
            row["p_holm"] = pa
        rows += fam
    df = pd.DataFrame(rows)
    if len(df):
        df["significant"] = df["p_holm"] < ALPHA
    return df


def cohen_dz(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    return float(d.mean() / sd) if sd and sd > 0 else np.nan


def tests_across_seeds(steps):
    """Seed-level paired tests of ΔF(FFT) - ΔF(LoRA).  Unit = (seed, step) pair."""
    rows = []
    for (model, order), g in steps[steps.step > 0].groupby(["model", "order"]):
        piv = g.pivot_table(index=["seed", "step"], columns="method", values="forgetting_pp")
        if not set(METHODS) <= set(piv.columns):
            continue
        piv = piv.dropna()
        for scope, sub in [("all steps", piv), ("step 3 only", piv.xs(3, level="step", drop_level=False))]:
            d = (sub["FFT"] - sub["LoRA"]).values
            row = {"model": model, "order": order, "scope": scope, "n_pairs": len(d),
                   "mean_diff_pp": round(d.mean(), 3) if len(d) else np.nan,
                   "sd_diff_pp": round(d.std(ddof=1), 3) if len(d) > 1 else np.nan,
                   "cohen_dz": round(cohen_dz(d), 3) if len(d) > 1 else np.nan,
                   "t_p": np.nan, "wilcoxon_p": np.nan}
            if len(d) >= 2 and np.std(d) > 0:
                row["t_p"] = float(stats.ttest_rel(sub["FFT"], sub["LoRA"]).pvalue)
            if len(d) >= 5 and np.any(d != 0):
                row["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def correlation(steps):
    rows = []
    for (model, order), g in steps[steps.step > 0].groupby(["model", "order"]):
        g = g.dropna(subset=["learning_gain", "forgetting_pp"])
        if len(g) < 4:
            continue
        rho, p = stats.spearmanr(g["learning_gain"], g["forgetting_pp"])
        rows.append({"model": model, "order": order, "n": len(g),
                     "spearman_rho": round(float(rho), 3), "p": float(p)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Hypothesis verdicts (rules frozen in Appendix A)
# ----------------------------------------------------------------------------------
def verdicts(steps, per_seed, corr):
    out = []
    for (model, order), g in steps.groupby(["model", "order"]):
        seeds = sorted(g.seed.unique())
        n_seeds = len(seeds)
        need = 2 if n_seeds >= 3 else n_seeds          # "≥2 of 3 seeds" (1 of 1 for single-seed checks)
        mean_f = g[g.step > 0].groupby(["method", "step"]).forgetting_pp.mean()
        ps = per_seed[(per_seed.model == model) & (per_seed.order == order)] if len(per_seed) else per_seed

        # H1 — forgetting appears after step 1 and grows with every step (checked per method)
        for meth in METHODS:
            if meth not in g.method.unique():
                continue
            f = [mean_f.get((meth, t), np.nan) for t in (1, 2, 3)]
            sig = ps[(ps.comparison == "baseline_vs_step") & (ps.method == meth)]
            sig3 = int(((sig.step == 3) & sig.significant & (sig.diff_pp > 0)).sum())
            sig_all = all(int(((sig.step == t) & sig.significant & (sig.diff_pp > 0)).sum()) >= need
                          for t in (1, 2, 3))
            monotonic = f[0] > 0 and f[0] < f[1] < f[2]
            if monotonic and sig_all:
                v = "supported"
            elif sig3 >= need:
                v = "partly supported"
            else:
                v = "not supported"
            out.append({"model": model, "order": order, "hypothesis": "H1", "method": meth,
                        "evidence": f"mean ΔF1..3 = {f[0]:+.2f}/{f[1]:+.2f}/{f[2]:+.2f} pp; monotonic={monotonic}; "
                                    f"step-3 Holm-significant drop in {sig3}/{n_seeds} seeds",
                        "verdict": v})

        # H2 — LoRA forgets less than FFT after every step
        if set(METHODS) <= set(g.method.unique()):
            diffs = [mean_f.get(("FFT", t), np.nan) - mean_f.get(("LoRA", t), np.nan) for t in (1, 2, 3)]
            sig = ps[ps.comparison == "LoRA_vs_FFT"]
            sig_t = {t: int(((sig.step == t) & sig.significant & (sig.diff_pp > 0)).sum()) for t in (1, 2, 3)}
            if all(d > 0 for d in diffs) and all(sig_t[t] >= need for t in (1, 2, 3)):
                v = "supported"
            elif sig_t[3] >= need and diffs[2] > 0:
                v = "partly supported"
            else:
                v = "not supported"
            out.append({"model": model, "order": order, "hypothesis": "H2", "method": "LoRA vs FFT",
                        "evidence": "mean ΔF(FFT)−ΔF(LoRA) per step = "
                                    + "/".join(f"{d:+.2f}" for d in diffs)
                                    + f" pp; Holm-significant LoRA advantage in {sig_t[1]}/{sig_t[2]}/{sig_t[3]} "
                                      f"of {n_seeds} seeds (steps 1/2/3)",
                        "verdict": v})

        # H3 — larger learning gain goes with larger forgetting
        c = corr[(corr.model == model) & (corr.order == order)] if len(corr) else corr
        if len(c):
            rho, p, n = float(c.spearman_rho.iloc[0]), float(c.p.iloc[0]), int(c.n.iloc[0])
            v = "supported" if rho > 0 and p < ALPHA else ("partly supported" if rho > 0 else "not supported")
            out.append({"model": model, "order": order, "hypothesis": "H3", "method": "both",
                        "evidence": f"Spearman ρ = {rho:+.2f}, p = {p:.3f}, n = {n} run-steps", "verdict": v})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------
# Diagnostics and error analysis
# ----------------------------------------------------------------------------------
def diagnostics(runs):
    rows = []
    for r in runs:
        for t in ("0", "1", "2", "3"):
            ev = r["eval"][t]
            cnt = Counter(ev["pred"])
            n = len(ev["pred"])
            shares = {f"share_{LETTERS[k]}": round(100 * cnt.get(k, 0) / n, 2) for k in range(4)}
            dom = max(cnt, key=cnt.get)
            rows.append({"model": model_label(r["model"]), "order": r["order"], "method": r["method"],
                         "seed": r["seed"], "step": int(t),
                         "mean_letter_mass": round(float(np.mean(ev["letter_mass"])), 4),
                         "median_letter_mass": round(float(np.median(ev["letter_mass"])), 4),
                         "dominant_letter": LETTERS[dom], "dominant_share": round(100 * cnt[dom] / n, 2),
                         **shares})
    return pd.DataFrame(rows)


def auto_category(m0, mt, pred_t, dom_letter, dom_share, probs_t):
    """First matching rule wins (frozen in Appendix A)."""
    if mt < 0.5 * m0:
        return "format_drift"
    if dom_share >= 0.40 and pred_t == dom_letter:
        return "letter_bias"
    top = sorted(probs_t, reverse=True)
    if len(top) > 1 and top[0] - top[1] < 0.10:
        return "near_tie"
    return "factual_erasure"


DEFINITIONS = {
    "format_drift": ("The model stops treating the question as multiple choice: probability on the answer "
                     "letters falls below half of its pre-training value.",
                     "SciQ/GSM8K/code train free-text answers, so the 'Answer:' slot is pulled toward words."),
    "letter_bias": ("The wrong answer is the letter the model now picks for at least 40% of all questions.",
                    "Fine-tuning shifts a prior toward one position instead of changing content knowledge."),
    "near_tie": ("Correct and chosen options are almost equally likely (gap < 0.10); the flip is marginal.",
                 "Small weight changes move borderline items; knowledge is weakened rather than lost."),
    "factual_erasure": ("Confident wrong answer with normal letter mass and no position bias.",
                        "Weights that stored the fact were overwritten by the new tasks."),
}


def error_analysis(runs, items, out):
    """Forgotten = correct at step 0, wrong at the final step."""
    smallest = min((model_size(r["model"]) for r in runs if r["order"] == "main"), default=None)
    if smallest is None:
        return pd.DataFrame(), {}
    chosen = [r for r in runs if r["order"] == "main" and model_size(r["model"]) == smallest
              and r["seed"] == ERROR_SEED]
    if not chosen:          # fall back to the first available seed
        seed0 = min(r["seed"] for r in runs if r["order"] == "main" and model_size(r["model"]) == smallest)
        chosen = [r for r in runs if r["order"] == "main" and model_size(r["model"]) == smallest
                  and r["seed"] == seed0]
    count_rows, samples = [], {}
    for r in chosen:
        e0, e3 = r["eval"]["0"], r["eval"]["3"]
        cnt = Counter(e3["pred"])
        dom = max(cnt, key=cnt.get)
        dom_share = cnt[dom] / len(e3["pred"])
        rows = []
        for i, it in enumerate(items):
            if not (e0["correct"][i] == 1 and e3["correct"][i] == 0):
                continue
            cat = auto_category(e0["letter_mass"][i], e3["letter_mass"][i], e3["pred"][i], dom, dom_share,
                                e3["probs"][i])
            rows.append({"item_id": it["id"], "source": it["src"], "gold": LETTERS[it["gold"]],
                         "pred_step0": LETTERS[e0["pred"][i]], "pred_step3": LETTERS[e3["pred"][i]],
                         "p_gold_step0": e0["probs"][i][it["gold"]], "p_gold_step3": e3["probs"][i][it["gold"]],
                         "letter_mass_step0": e0["letter_mass"][i], "letter_mass_step3": e3["letter_mass"][i],
                         "auto_category": cat, "manual_category": "", "note": "",
                         "question": it["prompt"]})
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        sample = df.sample(n=min(N_ERROR_SAMPLE, len(df)), random_state=42).sort_values("auto_category")
        sample.to_csv(os.path.join(out, f"error_sample_{r['method']}.csv"), index=False)
        samples[r["method"]] = sample
        for cat in DEFINITIONS:
            sub = df[df.auto_category == cat]
            ex = sub.iloc[0] if len(sub) else None
            count_rows.append({
                "method": r["method"], "model": model_label(r["model"]), "seed": r["seed"],
                "error_type": cat, "definition": DEFINITIONS[cat][0],
                "count": len(sub), "percent": round(100 * len(sub) / len(df), 1),
                "count_in_sample": int((sample.auto_category == cat).sum()),
                "example": (f"{ex.item_id}: gold {ex.gold}, step0 {ex.pred_step0} -> step3 {ex.pred_step3}, "
                            f"letter mass {ex.letter_mass_step0:.2f} -> {ex.letter_mass_step3:.2f}")
                if ex is not None else "",
                "possible_cause": DEFINITIONS[cat][1], "total_forgotten": len(df)})
    return pd.DataFrame(count_rows), samples


def efficiency(steps):
    s = steps[steps.step > 0]
    if s.empty:
        return pd.DataFrame()
    g = s.groupby(["model", "method"]).agg(
        gpu=("gpu", "first"), precision=("precision", "first"),
        train_time_per_step_min=("train_time_s", lambda x: round(x.mean() / 60, 2)),
        peak_vram_gb=("peak_vram_gb", "max"),
        trainable_params=("trainable_params", "first"), total_params=("total_params", "first"))
    g["trainable_percent"] = (100 * g.trainable_params / g.total_params).round(3)
    return g.reset_index()


# ----------------------------------------------------------------------------------
# Tables and figures
# ----------------------------------------------------------------------------------
def main_table(steps):
    def msd(x):
        x = x.dropna()
        return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}" if len(x) > 1 else (f"{x.mean():.2f}" if len(x) else "")
    rows = []
    for key, g in steps.groupby(["model", "order", "method", "step"]):
        rows.append({"model": key[0], "order": key[1], "method": key[2], "step": key[3],
                     "task": g.task.iloc[0], "n_seeds": g.seed.nunique(),
                     "old_acc": msd(g.old_acc), "forgetting_pp": msd(g.forgetting_pp),
                     "RFR_%": msd(g.RFR), "forgotten_items": msd(g.forgotten_items),
                     "gained_items": msd(g.gained_items), "learning_gain": msd(g.learning_gain),
                     "letter_mass": msd(g.letter_mass),
                     "old_acc_mean": g.old_acc.mean(), "forgetting_mean": g.forgetting_pp.mean()})
    return pd.DataFrame(rows)


def models_sorted(steps):
    return sorted(steps.model.unique(), key=lambda m: model_size("x/" + m))


def fig_trajectory(steps, out):
    s = steps[steps.order == "main"]
    models = models_sorted(s)
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.3), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        g = s[s.model == m]
        for meth in METHODS:
            h = g[g.method == meth].groupby("step").old_acc
            if h.size().empty:
                continue
            mu, sd = h.mean(), h.std(ddof=1).fillna(0)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=COLORS[meth], alpha=0.15, lw=0)
            ax.plot(mu.index, mu.values, marker="o", ms=5, lw=2, color=COLORS[meth],
                    label=f"{meth} (n={g[g.method == meth].seed.nunique()} seeds)")
            for _, r in g[g.method == meth].iterrows():
                ax.scatter(r.step, r.old_acc, s=8, color=COLORS[meth], alpha=0.5, lw=0)
        ax.axhline(25, color="#999999", ls=":", lw=1)
        ax.text(3, 25.8, "chance (4 options)", ha="right", va="bottom", fontsize=7, color="#666666")
        ax.set_xticks([0, 1, 2, 3])
        tasks = g.drop_duplicates("step").sort_values("step").task.tolist()
        ax.set_xticklabels(["base"] + tasks[1:], fontsize=8)
        ax.set_title(m)
        ax.set_xlabel("after training step")
    axes[0][0].set_ylabel("old-knowledge accuracy, % (ARC-C + MMLU)")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Fig. 1. Old-knowledge accuracy along the sequence (mean ± SD across seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_trajectory.png"))
    plt.close(fig)


def fig_forgetting(steps, out):
    s = steps[(steps.order == "main") & (steps.step > 0)]
    models = models_sorted(s)
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.3), sharey=True, squeeze=False)
    w = 0.36
    for ax, m in zip(axes[0], models):
        g = s[s.model == m]
        for j, meth in enumerate(METHODS):
            h = g[g.method == meth]
            if h.empty:
                continue
            mu = h.groupby("step").forgetting_pp.mean()
            sd = h.groupby("step").forgetting_pp.std(ddof=1).fillna(0)
            x = mu.index.values + (j - 0.5) * w
            ax.bar(x, mu.values, width=w - 0.04, color=COLORS[meth], label=meth, yerr=sd.values,
                   error_kw=dict(lw=1, capsize=2, ecolor="#555555"))
            for _, r in h.iterrows():
                ax.scatter(r.step + (j - 0.5) * w, r.forgetting_pp, s=10, color="white",
                           edgecolor="#333333", lw=0.6, zorder=3)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(g.drop_duplicates("step").sort_values("step").task.tolist())
        ax.set_title(m)
    axes[0][0].set_ylabel("forgetting ΔF, pp (baseline − current)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Fig. 2. Forgetting after each step (bars = mean, whiskers = SD, dots = seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_forgetting.png"))
    plt.close(fig)


def fig_tradeoff(steps, out):
    s = steps[(steps.order == "main") & (steps.step > 0)].dropna(subset=["learning_gain"])
    models = models_sorted(s)
    if not models:
        return
    marks = {1: "o", 2: "s", 3: "^"}
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.3), squeeze=False)
    for ax, m in zip(axes[0], models):
        g = s[s.model == m]
        for meth in METHODS:
            for t in (1, 2, 3):
                h = g[(g.method == meth) & (g.step == t)]
                ax.scatter(h.learning_gain, h.forgetting_pp, s=36, marker=marks[t], color=COLORS[meth],
                           edgecolor="white", lw=0.8, label=f"{meth}, step {t}")
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title(m)
        ax.set_xlabel("learning gain ΔL (held-out loss drop)")
    axes[0][0].set_ylabel("forgetting ΔF, pp")
    axes[0][0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Fig. 3. Learning gain vs forgetting (each point = one run-step)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_tradeoff.png"))
    plt.close(fig)


def fig_scale(steps, out):
    s = steps[(steps.order == "main") & (steps.step == 3)]
    models = models_sorted(s)
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    w = 0.36
    for j, meth in enumerate(METHODS):
        mus, sds = [], []
        for m in models:
            h = s[(s.model == m) & (s.method == meth)].forgetting_pp
            mus.append(h.mean())
            sds.append(h.std(ddof=1) if len(h) > 1 else 0)
        x = np.arange(len(models)) + (j - 0.5) * w
        ax.bar(x, mus, width=w - 0.04, color=COLORS[meth], yerr=sds, label=meth,
               error_kw=dict(lw=1, capsize=2, ecolor="#555555"))
        for xi, m in zip(x, models):
            for v in s[(s.model == m) & (s.method == meth)].forgetting_pp:
                ax.scatter(xi, v, s=10, color="white", edgecolor="#333333", lw=0.6, zorder=3)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("forgetting after step 3, pp")
    ax.legend(fontsize=8)
    ax.set_title("Fig. 4. Does a bigger model forget less?")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_scale.png"))
    plt.close(fig)


def fig_diagnostics(diag, out):
    d = diag[diag.order == "main"]
    models = models_sorted(d)
    if not models:
        return
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 5.6), squeeze=False)
    for col, m in enumerate(models):
        g = d[d.model == m]
        ax = axes[0][col]
        for meth in METHODS:
            h = g[g.method == meth].groupby("step").mean_letter_mass
            if h.size().empty:
                continue
            ax.plot(h.mean().index, h.mean().values, marker="o", lw=2, ms=5, color=COLORS[meth], label=meth)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["base", "1", "2", "3"])
        ax.set_title(f"{m}: probability on answer letters")
        ax.set_ylabel("mean letter mass")
        ax = axes[1][col]
        w = 0.36
        for j, meth in enumerate(METHODS):
            h = g[(g.method == meth) & (g.step == 3)]
            if h.empty:
                continue
            vals = [h[f"share_{L}"].mean() for L in LETTERS[:4]]
            ax.bar(np.arange(4) + (j - 0.5) * w, vals, width=w - 0.04, color=COLORS[meth], label=f"{meth}, step 3")
        base = g[g.step == 0]
        if len(base):
            ax.scatter(np.arange(4), [base[f"share_{L}"].mean() for L in LETTERS[:4]], marker="_", s=300,
                       color="#222222", lw=2, zorder=3, label="base model")
        ax.set_xticks(range(4))
        ax.set_xticklabels(LETTERS[:4])
        ax.set_ylabel("% of questions answered with letter")
        ax.set_title(f"{m}: which letter the model picks")
    axes[0][0].legend(fontsize=8)
    axes[1][0].legend(fontsize=8)
    fig.suptitle("Fig. 5. Format diagnostics: does the model still answer with a letter, and which one?", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_diagnostics.png"))
    plt.close(fig)


def fig_order(steps, out):
    rev = steps[steps.order == "reversed"]
    if rev.empty:
        return
    m = rev.model.iloc[0]
    seeds = set(rev[rev.model == m].seed)
    main = steps[(steps.order == "main") & (steps.model == m) & (steps.seed.isin(seeds))]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for meth in METHODS:
        for order, ls in (("main", "-"), ("reversed", "--")):
            h = steps[(steps.model == m) & (steps.order == order) & (steps.method == meth) &
                      (steps.seed.isin(seeds))].groupby("step").old_acc.mean()
            if h.empty:
                continue
            ax.plot(h.index, h.values, ls=ls, marker="o", ms=4, lw=2, color=COLORS[meth],
                    label=f"{meth}, {order} order")
    if main.empty:
        pass
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["base", "1", "2", "3"])
    ax.set_xlabel("after training step")
    ax.set_ylabel("old-knowledge accuracy, %")
    ax.set_title(f"Fig. 6. Task order check ({m})\nmain: GSM8K→Code→SciQ, reversed: SciQ→Code→GSM8K", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_order.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------
def md_table(df, cols=None):
    if df is None or len(df) == 0:
        return "_no data yet_\n"
    df = df[cols] if cols else df
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def write_summary(out, runs, items_sha, table, per_seed, across, corr, verd, diag, err, eff):
    L = ["# Assignment 4 — results digest (auto-generated by analysis.py)\n"]
    L.append(f"Runs found: **{len(runs)}**  ")
    L.append(f"Evaluation set: {len(set(r['eval_items_sha256'] for r in runs))} distinct hash(es); "
             f"eval_items.json sha256 = `{items_sha}`\n")
    inv = pd.DataFrame([{"model": model_label(r["model"]), "order": r["order"], "method": r["method"],
                         "seed": r["seed"], "gpu": r["environment"]["gpu"], "precision": r.get("precision", "")}
                        for r in runs]).sort_values(["model", "order", "method", "seed"])
    L.append("## Runs\n" + md_table(inv))
    L.append("## Main table (mean ± SD across seeds)\n")
    L.append(md_table(table, ["model", "order", "method", "step", "task", "n_seeds", "old_acc",
                              "forgetting_pp", "RFR_%", "forgotten_items", "gained_items",
                              "learning_gain", "letter_mass"]))
    L.append("## Hypothesis verdicts (frozen decision rules)\n" + md_table(verd))
    L.append("## Seed-level test: ΔF(FFT) − ΔF(LoRA)\n" + md_table(across))
    L.append("## Learning gain vs forgetting (H3)\n" + md_table(corr))
    if len(per_seed):
        p3 = per_seed[per_seed.step == 3][["comparison", "model", "order", "method", "seed", "diff_pp",
                                          "ci_low", "ci_high", "p", "p_holm", "significant"]]
        L.append("## Per-seed tests at step 3 (McNemar exact, bootstrap 95% CI, Holm)\n" + md_table(p3))
    L.append("## Error analysis (Appendix B; auto categories, check manually)\n")
    L.append(md_table(err, ["method", "error_type", "count", "percent", "count_in_sample", "example"])
             if len(err) else "_no data yet_\n")
    d3 = diag[diag.step.isin([0, 3])].groupby(["model", "order", "method", "step"]).agg(
        mean_letter_mass=("mean_letter_mass", "mean"), dominant_share=("dominant_share", "mean")).reset_index() \
        if len(diag) else diag
    L.append("## Format diagnostics (base vs final step)\n" + md_table(d3))
    L.append("## Efficiency\n" + md_table(eff))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results"])
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load_runs(a.results)
    if not runs:
        raise SystemExit(f"no run files found in {a.results} (expected <folder>/runs/*.json)")
    items, items_sha = load_items(a.results)
    hashes = {r["eval_items_sha256"] for r in runs}
    if len(hashes) > 1:
        print(f"WARNING: runs were evaluated on {len(hashes)} different item sets: {hashes}")

    steps = per_run_steps(runs)
    steps.to_csv(os.path.join(a.out, "per_run_steps.csv"), index=False)
    table = main_table(steps)
    table.to_csv(os.path.join(a.out, "table_main.csv"), index=False)
    per_seed = tests_per_seed(runs)
    per_seed.to_csv(os.path.join(a.out, "tests_per_seed.csv"), index=False)
    across = tests_across_seeds(steps)
    across.to_csv(os.path.join(a.out, "tests_across_seeds.csv"), index=False)
    corr = correlation(steps)
    corr.to_csv(os.path.join(a.out, "correlation.csv"), index=False)
    verd = verdicts(steps, per_seed, corr)
    verd.to_csv(os.path.join(a.out, "verdicts.csv"), index=False)
    diag = diagnostics(runs)
    diag.to_csv(os.path.join(a.out, "diagnostics.csv"), index=False)
    err, _ = error_analysis(runs, items, a.out)
    err.to_csv(os.path.join(a.out, "error_counts.csv"), index=False)
    eff = efficiency(steps)
    eff.to_csv(os.path.join(a.out, "efficiency.csv"), index=False)

    fig_trajectory(steps, a.out)
    fig_forgetting(steps, a.out)
    fig_tradeoff(steps, a.out)
    fig_scale(steps, a.out)
    fig_diagnostics(diag, a.out)
    fig_order(steps, a.out)
    write_summary(a.out, runs, items_sha, table, per_seed, across, corr, verd, diag, err, eff)

    print(f"{len(runs)} runs analysed -> {a.out}/")
    print(verd.to_string(index=False) if len(verd) else "no verdicts")


if __name__ == "__main__":
    main()
