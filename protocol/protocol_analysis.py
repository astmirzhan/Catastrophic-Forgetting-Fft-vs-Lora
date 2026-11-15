"""
Tables, figure and checks for the evaluation-protocol question (see protocol_eval.py).

    python protocol/protocol_analysis.py --res results_protocol --out report_protocol
    python protocol/protocol_analysis.py --res results_protocol --verify letter_shuf FFT 42 ARC

Definitions (per evaluation mode and subset ARC / MMLU / All):
    accuracy     letter_orig, full_answer: % correct over the items.
                 letter_shuf: % correct averaged over the 5 shuffles (= % over all item x shuffle rows).
    forgetting   Step 0 accuracy - Step 3 accuracy (pp); Step 0 is the same base model for both methods.
    method gap   FFT forgetting - LoRA forgetting (pp)  ( = LoRA Step-3 acc - FFT Step-3 acc ).
    gap change   gap(mode) - gap(letter_orig).
95% CIs: paired bootstrap over the 1,000 items (an item is resampled together with all its
shuffles and with all checkpoints), 2,000 resamples, seed 0.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

MODES = ["letter_orig", "letter_shuf", "full_answer"]
LABEL = {"letter_orig": "Letter, original order", "letter_shuf": "Letter, 5 shuffles",
         "full_answer": "Full-answer log-prob"}
SUBSETS = {"ARC": "ARC-Challenge", "MMLU": "MMLU", "All": None}
N_BOOT = 2000


def load(res):
    files = sorted(glob.glob(os.path.join(res, "items", "*.csv")))
    assert files, f"no item CSVs in {res}/items"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def matched_seeds(df):
    s3 = df[df.step == 3]
    have = s3.groupby("seed").method.unique()
    return sorted(int(s) for s, m in have.items() if {"FFT", "LoRA"} <= set(m))


def item_matrix(df, mode, subset):
    """items x checkpoints matrix of per-item accuracy (mean over shuffles for letter_shuf)."""
    d = df[df["mode"] == mode]
    if SUBSETS[subset]:
        d = d[d.src == SUBSETS[subset]]
    return d.pivot_table(index="item_id", columns="ckpt", values="correct", aggfunc="mean").sort_index()


def present(df):
    return [s for s, v in SUBSETS.items() if v is None or (df.src == v).any()]


def ck3(method, seed):
    return f"qwen0.5b_main_{method}_s{seed}_step3"


CK0 = "qwen0.5b_step0"


def results_table(df, seeds):
    rows = []
    for mode in MODES:
        for sub in present(df):
            M = item_matrix(df, mode, sub)
            a0 = 100 * M[CK0].mean()
            for seed in seeds:
                for method in ["FFT", "LoRA"]:
                    a3 = 100 * M[ck3(method, seed)].mean()
                    r = {"mode": mode, "subset": sub, "seed": seed, "method": method,
                         "n_items": len(M), "step0_acc": round(a0, 2), "step3_acc": round(a3, 2),
                         "forgetting_pp": round(a0 - a3, 2)}
                    if mode == "letter_shuf":
                        d = df[(df["mode"] == mode) & (df.ckpt == ck3(method, seed))]
                        if SUBSETS[sub]:
                            d = d[d.src == SUBSETS[sub]]
                        per = 100 * d.groupby("shuffle").correct.mean()
                        r["step3_acc_shuffle_min"] = round(per.min(), 2)
                        r["step3_acc_shuffle_max"] = round(per.max(), 2)
                    rows.append(r)
    return pd.DataFrame(rows)


def gaps(df, seeds):
    rng = np.random.default_rng(0)
    rows = []
    for sub in present(df):
        mats = {m: item_matrix(df, m, sub) for m in MODES}
        idx = mats["letter_orig"].index
        n = len(idx)
        boot = rng.integers(0, n, size=(N_BOOT, n))
        for seed in seeds:
            g, gb = {}, {}
            for mode in MODES:
                M = mats[mode].loc[idx]
                f = M[ck3("FFT", seed)].to_numpy()
                l = M[ck3("LoRA", seed)].to_numpy()
                g[mode] = 100 * (l.mean() - f.mean())
                gb[mode] = 100 * (l[boot].mean(1) - f[boot].mean(1))
            for mode in MODES:
                lo, hi = np.percentile(gb[mode], [2.5, 97.5])
                r = {"subset": sub, "seed": seed, "mode": mode, "gap_pp": round(g[mode], 2),
                     "gap_ci_lo": round(lo, 2), "gap_ci_hi": round(hi, 2)}
                if mode != "letter_orig":
                    ch = gb[mode] - gb["letter_orig"]
                    clo, chi = np.percentile(ch, [2.5, 97.5])
                    r.update({"change_vs_orig_pp": round(g[mode] - g["letter_orig"], 2),
                              "change_ci_lo": round(clo, 2), "change_ci_hi": round(chi, 2)})
                rows.append(r)
    return pd.DataFrame(rows)


def gap_summary(G):
    out = []
    for (sub, mode), d in G.groupby(["subset", "mode"], sort=False):
        r = {"subset": sub, "mode": mode, "n_seeds": len(d),
             "gap_mean": round(d.gap_pp.mean(), 2),
             "gap_sd": round(d.gap_pp.std(ddof=1), 2) if len(d) > 1 else np.nan,
             "gap_per_seed": " / ".join(f"{v:.1f}" for v in d.gap_pp)}
        if mode != "letter_orig":
            r["change_mean"] = round(d.change_vs_orig_pp.mean(), 2)
            r["change_per_seed"] = " / ".join(f"{v:+.1f}" for v in d.change_vs_orig_pp)
            r["seeds_with_ci_excluding_0"] = int(((d.change_ci_lo > 0) | (d.change_ci_hi < 0)).sum())
        out.append(r)
    return pd.DataFrame(out)


def position_table(df, seeds):
    """letter_shuf: accuracy by position of the correct answer, and share of predictions per letter."""
    d = df[(df["mode"] == "letter_shuf") & (df.n_opt == 4)]
    rows = []
    cks = [(CK0, "base", "")] + [(ck3(m, s), m, s) for s in seeds for m in ["FFT", "LoRA"]]
    for ck, method, seed in cks:
        x = d[d.ckpt == ck]
        for pos in range(4):
            rows.append({"ckpt": ck, "method": method, "seed": seed, "letter": "ABCD"[pos],
                         "acc_when_gold_here": round(100 * x[x.gold_pos == pos].correct.mean(), 2),
                         "n_gold_here": int((x.gold_pos == pos).sum()),
                         "pred_share": round(100 * (x.pred_pos == pos).mean(), 2)})
    return pd.DataFrame(rows)


def reproduction(res):
    rows = []
    for f in sorted(glob.glob(os.path.join(res, "repro", "*.json"))):
        d = json.load(open(f))
        r = {"ckpt": os.path.basename(f)[:-5], **d["vs_a4_log"]}
        for s, v in d.get("steps_1_2_vs_log", {}).items():
            r[f"step{s}_diff_pp"] = v["diff_pp"]
        rows.append(r)
    return pd.DataFrame(rows)


def figure(G, seeds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    x = np.arange(len(MODES))
    marks = ["o", "s", "^", "D", "v"]
    for ax, sub in zip(axes, ["All", "ARC", "MMLU"]):
        d = G[G.subset == sub]
        if d.empty:
            continue
        means = [d[d["mode"] == m].gap_pp.mean() for m in MODES]
        ax.bar(x, means, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.35, width=0.6,
               label="mean over seeds")
        for k, s in enumerate(seeds):
            ds = d[d.seed == s].set_index("mode").loc[MODES]
            ax.errorbar(x + (k - (len(seeds) - 1) / 2) * 0.12, ds.gap_pp,
                        yerr=[ds.gap_pp - ds.gap_ci_lo, ds.gap_ci_hi - ds.gap_pp],
                        fmt=marks[k % 5] + "-", color="0.15", ms=4, lw=0.8, capsize=2,
                        label=f"seed {s} (95% CI)")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Letter\noriginal", "Letter\n5 shuffles", "Full-answer\nlog-prob"])
        ax.set_title({"All": "ARC + MMLU (1,000 items)", "ARC": "ARC-Challenge", "MMLU": "MMLU"}[sub])
    axes[0].set_ylabel("FFT − LoRA forgetting at Step 3, pp")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verify(df, mode, method, seed, subset):
    """Recount one table row straight from the item-level CSV rows, without pandas."""
    want = SUBSETS[subset]
    out = {}
    for step, ck in [(0, CK0), (3, ck3(method, seed))]:
        n = c = 0
        for r in df[(df.ckpt == ck) & (df["mode"] == mode)].itertuples():
            if want is None or r.src == want:
                n += 1
                c += int(r.correct)
        out[step] = (c, n)
    a0 = 100 * out[0][0] / out[0][1]
    a3 = 100 * out[3][0] / out[3][1]
    print(f"{mode} {method} seed {seed} {subset}: Step 0 {out[0][0]}/{out[0][1]} = {a0:.2f}%, "
          f"Step 3 {out[3][0]}/{out[3][1]} = {a3:.2f}%, forgetting {a0 - a3:.2f} pp")
    print("(letter_shuf counts are item x shuffle rows, so n = 5 x items)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_protocol")
    ap.add_argument("--out", default="report_protocol")
    ap.add_argument("--verify", nargs=4, metavar=("MODE", "METHOD", "SEED", "SUBSET"))
    a = ap.parse_args()
    df = load(a.res)
    if a.verify:
        verify(df, a.verify[0], a.verify[1], int(a.verify[2]), a.verify[3])
        return
    os.makedirs(a.out, exist_ok=True)
    seeds = matched_seeds(df)
    print(f"matched seeds (both FFT and LoRA at Step 3): {seeds}")
    T = results_table(df, seeds)
    G = gaps(df, seeds)
    S = gap_summary(G)
    P = position_table(df, seeds)
    R = reproduction(a.res)
    T.to_csv(os.path.join(a.out, "table_protocol.csv"), index=False)
    G.to_csv(os.path.join(a.out, "gap_per_seed.csv"), index=False)
    S.to_csv(os.path.join(a.out, "gap_summary.csv"), index=False)
    P.to_csv(os.path.join(a.out, "position_shuffled.csv"), index=False)
    R.to_csv(os.path.join(a.out, "reproduction.csv"), index=False)
    figure(G, seeds, os.path.join(a.out, "fig_gap_by_mode.png"))
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(f"Matched seeds: {seeds}\n\n## Results table\n\n")
        f.write(T.to_markdown(index=False))
        f.write("\n\n## Gap summary\n\n" + S.to_markdown(index=False))
        f.write("\n\n## Gap per seed (95% bootstrap CI)\n\n" + G.to_markdown(index=False))
        f.write("\n\n## Reproduction of Assignment 4 logs (letter_orig)\n\n" + R.to_markdown(index=False))
        f.write("\n\n## Shuffled options: accuracy by position of the correct answer\n\n" + P.to_markdown(index=False))
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    print(R.to_string(index=False))
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
