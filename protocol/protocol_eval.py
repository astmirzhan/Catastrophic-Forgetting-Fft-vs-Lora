"""
Evaluation-protocol check for the Assignment 4 FFT-vs-LoRA gap (Qwen2.5-0.5B, main order).

Question: does the 10.8 pp FFT-LoRA forgetting gap at Step 3 survive a different scoring rule
or shuffled answer options?

Three evaluation modes, all on the frozen 1,000 ARC-Challenge + MMLU items
(SHA-256 e6e4c60c...4bb9224cf1):

  letter_orig   original prompt and option order, prediction = answer letter with the highest
                next-token probability (exactly forgetting.eval_old_knowledge, imported).
  letter_shuf   the same letter scoring on 5 fixed option shuffles per item. The correct letter
                is remapped after each shuffle. Permutations are generated once from a fixed
                seed, saved to shuffles.json and reused for every checkpoint.
  full_answer   original prompt and option order; every candidate's answer text is appended
                after "Answer:" and scored by its mean log-probability per token; the candidate
                with the highest score is the prediction.

Checkpoints. The Assignment 4 runs did not save weights (save_strategy="no"), so Step 3 is
re-trained here with the frozen code path of forgetting.py (same functions, same order of
operations, same seed). Step 0 is the untouched base model. The original letter score of
every re-trained checkpoint is compared item by item with the Assignment 4 run logs, so the
reduced exactness (GPU non-determinism) is measured and reported, not assumed.

Every checkpoint writes <out>/items/<ckpt>.csv (one row per item x mode x shuffle) and
<out>/repro/<ckpt>.json. A checkpoint whose CSV already exists is skipped, so the command
can simply be restarted after a Colab disconnect.

    python protocol/protocol_eval.py --out results_protocol --seeds 42 43 44
    python protocol/protocol_eval.py --out results_protocol --seeds 42          # reduced coverage
    python protocol/protocol_eval.py --out results_protocol --seeds 42 --save-weights /content/drive/MyDrive/rtm_ckpt
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import forgetting as F  # noqa: E402  (frozen Assignment 4 code, not modified)

FROZEN_SHA = "e6e4c60c9f2b95a488bda1388975af49be4204bfc9b0c139e0b35a4bb9224cf1"
N_SHUFFLES = 5
SHUFFLE_SEED = 20261116            # fixed before any protocol-check result was seen
MODEL_ID = "Qwen/Qwen2.5-0.5B"
ORDER = "main"
FIELDS = ["ckpt", "model", "method", "seed", "step", "mode", "shuffle", "item_id", "src",
          "n_opt", "perm", "gold_orig", "gold_pos", "pred_pos", "pred_orig", "correct",
          "letter_mass", "score_gold", "score_pred"]


# ----------------------------------------------------------------------------------
# Items and shuffles
# ----------------------------------------------------------------------------------
def load_items(path):
    with open(path) as f:
        d = json.load(f)
    items = d["items"]
    sha = F.items_hash(items)
    assert sha == FROZEN_SHA, f"eval items hash {sha} != frozen {FROZEN_SHA}"
    for it in items:
        it["question"], it["choices"] = parse_prompt(it["prompt"], it["n_opt"])
    return items


def parse_prompt(prompt, n_opt):
    """Recover question and option texts from the frozen prompt and prove it by rebuilding."""
    assert prompt.startswith("Question: ") and prompt.endswith("\nAnswer:")
    body = prompt[len("Question: "):-len("Answer:")]           # "...q...\nA. x\nB. y\n"
    choices = []
    for k in reversed(range(n_opt)):
        tag = f"\n{F.LETTERS[k]}. "
        i = body.rfind(tag)
        assert i >= 0, f"option {F.LETTERS[k]} not found"
        choices.append(body[i + len(tag):].removesuffix("\n"))
        body = body[:i + 1]
    question = body.removesuffix("\n")
    choices = choices[::-1]
    assert F.mc_prompt(question, choices) == prompt, "prompt could not be rebuilt exactly"
    return question, choices


def make_shuffles(items, path):
    """5 distinct non-identity permutations per item; saved once, reused for every checkpoint."""
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        assert d["seed"] == SHUFFLE_SEED and d["items_sha256"] == FROZEN_SHA
        return d["perms"]
    rng = np.random.default_rng(SHUFFLE_SEED)
    perms = {}
    for it in items:
        n = it["n_opt"]
        ident, chosen = tuple(range(n)), []
        while len(chosen) < N_SHUFFLES:
            p = tuple(int(x) for x in rng.permutation(n))
            if p != ident and p not in chosen:
                chosen.append(p)
        perms[it["id"]] = [list(p) for p in chosen]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": SHUFFLE_SEED, "items_sha256": FROZEN_SHA, "perms": perms}, f)
    return perms


def shuffled_items(items, perms):
    """perm[j] = original index of the option shown at position j."""
    out = []
    for k in range(N_SHUFFLES):
        for it in items:
            p = perms[it["id"]][k]
            out.append({"id": it["id"], "src": it["src"], "n_opt": it["n_opt"], "shuffle": k + 1,
                        "perm": p, "gold_orig": it["gold"], "gold": p.index(it["gold"]),
                        "prompt": F.mc_prompt(it["question"], [it["choices"][j] for j in p])})
    return out


# ----------------------------------------------------------------------------------
# Full-answer scoring
# ----------------------------------------------------------------------------------
@torch.no_grad()
def eval_full_answer(model, tok, items, items_per_batch=4):
    """Mean log-probability per token of each candidate answer text after the original prompt.

    Context and continuation are tokenised separately and concatenated, so the continuation
    tokens are exactly the tokens of " <answer text>". Only continuation positions go through
    the LM head (full-vocabulary logits for every position would not fit on a T4).
    """
    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id
    pred, correct, scores = [], [], []
    for start in range(0, len(items), items_per_batch):
        batch = items[start:start + items_per_batch]
        seqs, spans, owner = [], [], []
        for b, it in enumerate(batch):
            ctx = tok(it["prompt"], add_special_tokens=False)["input_ids"]
            for c in it["choices"]:
                cont = tok(" " + c.strip(), add_special_tokens=False)["input_ids"]
                seqs.append(ctx + cont)
                spans.append((len(ctx), len(ctx) + len(cont)))
                owner.append(b)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):                          # right padding
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(device), att.to(device)
        hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        sc = []
        for i, (a, e) in enumerate(spans):
            if e == a:                                       # empty answer text
                sc.append(float("-inf"))
                continue
            h = hidden[i, a - 1:e - 1, :]                    # positions that predict cont tokens
            lp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
            tgt = ids[i, a:e]
            sc.append(float(lp.gather(1, tgt[:, None]).mean()))
        k0 = 0
        for b, it in enumerate(batch):
            s = sc[k0:k0 + it["n_opt"]]
            k0 += it["n_opt"]
            k = int(np.argmax(s))
            pred.append(k)
            correct.append(int(k == it["gold"]))
            scores.append([round(v, 5) for v in s])
    return {"pred": pred, "correct": correct, "scores": scores}


# ----------------------------------------------------------------------------------
# One checkpoint -> item-level rows
# ----------------------------------------------------------------------------------
def evaluate_checkpoint(model, tok, letter_ids, items, shuf, meta):
    rows, summary = [], {}
    t0 = time.time()
    ev = F.eval_old_knowledge(model, tok, items, letter_ids)
    for it, p, c, m, pr in zip(items, ev["pred"], ev["correct"], ev["letter_mass"], ev["probs"]):
        rows.append({**meta, "mode": "letter_orig", "shuffle": 0, "item_id": it["id"], "src": it["src"],
                     "n_opt": it["n_opt"], "perm": "", "gold_orig": it["gold"], "gold_pos": it["gold"],
                     "pred_pos": p, "pred_orig": p, "correct": c, "letter_mass": m,
                     "score_gold": pr[it["gold"]], "score_pred": pr[p]})
    summary["letter_orig"] = 100 * float(np.mean(ev["correct"]))
    F.log(f"  letter_orig  {summary['letter_orig']:.2f}%  ({time.time() - t0:.0f}s)")
    orig_eval = ev

    t0 = time.time()
    ev = F.eval_old_knowledge(model, tok, shuf, letter_ids)
    for it, p, c, m, pr in zip(shuf, ev["pred"], ev["correct"], ev["letter_mass"], ev["probs"]):
        rows.append({**meta, "mode": "letter_shuf", "shuffle": it["shuffle"], "item_id": it["id"],
                     "src": it["src"], "n_opt": it["n_opt"], "perm": "".join(map(str, it["perm"])),
                     "gold_orig": it["gold_orig"], "gold_pos": it["gold"], "pred_pos": p,
                     "pred_orig": it["perm"][p], "correct": c, "letter_mass": m,
                     "score_gold": pr[it["gold"]], "score_pred": pr[p]})
    summary["letter_shuf"] = 100 * float(np.mean(ev["correct"]))
    F.log(f"  letter_shuf  {summary['letter_shuf']:.2f}% (mean of {N_SHUFFLES})  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    ev = eval_full_answer(model, tok, items)
    for it, p, c, s in zip(items, ev["pred"], ev["correct"], ev["scores"]):
        rows.append({**meta, "mode": "full_answer", "shuffle": 0, "item_id": it["id"], "src": it["src"],
                     "n_opt": it["n_opt"], "perm": "", "gold_orig": it["gold"], "gold_pos": it["gold"],
                     "pred_pos": p, "pred_orig": p, "correct": c, "letter_mass": "",
                     "score_gold": s[it["gold"]], "score_pred": s[p]})
    summary["full_answer"] = 100 * float(np.mean(ev["correct"]))
    F.log(f"  full_answer  {summary['full_answer']:.2f}%  ({time.time() - t0:.0f}s)")
    return rows, summary, orig_eval



def write_rows(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def pick(ev, sel):
    return {k: [v[i] for i in sel] for k, v in ev.items()}


def compare_with_log(ev, logged):
    """Item-level agreement between a fresh letter_orig evaluation and the Assignment 4 log."""
    a, b = np.array(ev["pred"]), np.array(logged["pred"])
    ca, cb = np.array(ev["correct"]), np.array(logged["correct"])
    return {"acc_now": round(float(100 * ca.mean()), 2), "acc_logged": round(float(100 * cb.mean()), 2),
            "diff_pp": round(float(100 * (ca.mean() - cb.mean())), 2),
            "same_prediction_pct": round(100 * float((a == b).mean()), 2),
            "max_abs_letter_mass_diff": round(float(np.max(np.abs(
                np.array(ev["letter_mass"]) - np.array(logged["letter_mass"])))), 6)}


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--methods", nargs="+", default=["FFT", "LoRA"], choices=["FFT", "LoRA"])
    ap.add_argument("--a4-results", default="results_colab", help="Assignment 4 results (logs to reproduce)")
    ap.add_argument("--out", default="results_protocol")
    ap.add_argument("--workdir", default="/tmp/rtm_ckpt")
    ap.add_argument("--save-weights", default=None, help="directory to save re-trained Step-3 weights")
    ap.add_argument("--load-weights", default=None, help="directory with saved Step-3 weights (skip training)")
    ap.add_argument("--step0-only", action="store_true")
    ap.add_argument("--match-t4", action="store_true",
                    help="train with the Assignment 4 Colab-T4 recipe (fp16 mixed, batch 2 x 16) on any GPU")
    ap.add_argument("--max-items", type=int, default=None, help="debug only: evaluate the first N items")
    a = ap.parse_args()

    if a.match_t4 and torch.cuda.is_available():
        # forgetting.py would pick bf16 and batch 8 x 4 on a >30 GB / Ampere GPU; the 0.5B
        # Assignment 4 runs used fp16 mixed precision and batch 2 x 16 on a T4.
        F.precision = lambda: (False, True)
        F.per_device_batch = lambda: (2, 16)
        F.log("match-t4: fp16 mixed precision, per-device batch 2 x 16")
    os.makedirs(os.path.join(a.out, "items"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "repro"), exist_ok=True)
    F.save_environment(a.out)
    items = load_items(os.path.join(a.a4_results, "eval_items.json"))
    perms = make_shuffles(items, os.path.join(a.out, "shuffles.json"))
    sel = list(range(len(items)))
    if a.max_items:                    # debug: first N/2 ARC + first N/2 MMLU items
        k = a.max_items // 2
        sel = list(range(k)) + list(range(500, 500 + k))
        items = [items[i] for i in sel]
    shuf = shuffled_items(items, perms)
    src = np.array([it["src"] for it in items])
    short = F.short_name(a.model)
    F.log(f"items {len(items)} | shuffled prompts {len(shuf)} | env {F.environment()}")

    tok = F.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = F.letter_token_ids(tok)

    # ---------------- Step 0 (base model, shared by both methods) ----------------
    ck0 = f"{short}_step0"
    p0 = os.path.join(a.out, "items", f"{ck0}.csv")
    if os.path.exists(p0):
        F.log(f"skip {ck0} (already done)")
    else:
        F.log(f"=== {ck0} ===")
        model = F.load_base_model(a.model)
        meta = {"ckpt": ck0, "model": a.model, "method": "base", "seed": "", "step": 0}
        rows, summ, ev = evaluate_checkpoint(model, tok, letter_ids, items, shuf, meta)
        with open(os.path.join(a.a4_results, "baselines", f"{short}.json")) as f:
            logged = json.load(f)["eval"]
        rep = compare_with_log(ev, pick(logged, sel))
        F.log(f"  reproduction of logged Step 0: {rep}")
        with open(os.path.join(a.out, "repro", f"{ck0}.json"), "w") as f:
            json.dump({"summary": summ, "vs_a4_log": rep, "env": F.environment()}, f, indent=2)
        write_rows(p0, rows)
        del model
        F.free_gpu()
    if a.step0_only:
        return

    # ---------------- Step 3 (re-trained with the frozen code path) ----------------
    cfg = dict(F.FROZEN)
    tasks = None
    for seed in a.seeds:
        for method in a.methods:
            ck = f"{short}_{ORDER}_{method}_s{seed}_step3"
            path = os.path.join(a.out, "items", f"{ck}.csv")
            if os.path.exists(path):
                F.log(f"skip {ck} (already done)")
                continue
            F.log(f"=== {ck} ===")
            run_id = f"{short}_{ORDER}_{method}_s{seed}"
            with open(os.path.join(a.a4_results, "runs", f"{run_id}.json")) as f:
                logged = json.load(f)
                step_rep = {}
            wdir = os.path.join(a.load_weights, ck) if a.load_weights else None
            if wdir and os.path.isdir(wdir):
                F.log(f"  loading weights from {wdir}")
                model = F.AutoModelForCausalLM.from_pretrained(wdir).float()
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                trained = False
            else:
                if tasks is None:
                    tasks = [F.load_task(name, tok, cfg) for name in F.ORDERS[ORDER]]
                model = retrain(a.model, method, seed, tasks, items, letter_ids, cfg, a.workdir,
                                logged, sel, step_rep)
                trained = True
            meta = {"ckpt": ck, "model": a.model, "method": method, "seed": seed, "step": 3}
            rows, summ, ev = evaluate_checkpoint(model, tok, letter_ids, items, shuf, meta)
            rep = compare_with_log(ev, pick(logged["eval"]["3"], sel))
            F.log(f"  reproduction of logged Step 3: {rep}")
            with open(os.path.join(a.out, "repro", f"{ck}.json"), "w") as f:
                json.dump({"summary": summ, "vs_a4_log": rep, "steps_1_2_vs_log": step_rep,
                           "retrained": trained, "match_t4": a.match_t4,
                           "precision": "fp16" if F.precision()[1] else ("bf16" if F.precision()[0] else "fp32"),
                           "batch_x_accum": list(F.per_device_batch()), "env": F.environment()}, f, indent=2)
            if a.save_weights and trained:
                d = os.path.join(a.save_weights, ck)
                model.save_pretrained(d, safe_serialization=True)
                tok.save_pretrained(d)
                F.log(f"  weights saved to {d}")
            write_rows(path, rows)
            del model
            F.free_gpu()
    F.log("all requested checkpoints finished")


def retrain(model_id, method, seed, tasks, items, letter_ids, cfg, workdir, logged, sel, step_rep):
    """Same operations, in the same order, as forgetting.run_one (which does not return the model)."""
    F.set_seed(seed)
    tok = F.AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = F.load_base_model(model_id)
    for t, task in enumerate(tasks, start=1):
        F.heldout_loss(model, task["heldout"])
        if method == "LoRA":
            L = cfg["lora"]
            model = F.get_peft_model(model, F.LoraConfig(task_type=F.TaskType.CAUSAL_LM, r=L["r"],
                                                         lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                                                         target_modules=L["modules"]))
        _, _, dt, _ = F.train_one_step(model, method, task, tok, cfg, seed, workdir)
        F.free_gpu()
        if method == "LoRA":
            model = model.merge_and_unload()
        F.heldout_loss(model, task["heldout"])
        if t < len(tasks):          # Step 3 is evaluated (and compared) by the caller
            ev = F.eval_old_knowledge(model, tok, items, letter_ids)
            step_rep[str(t)] = compare_with_log(ev, pick(logged["eval"][str(t)], sel))
            F.log(f"  step {t} {task['name']} trained in {dt:.0f}s; vs log: {step_rep[str(t)]}")
        else:
            F.log(f"  step {t} {task['name']} trained in {dt:.0f}s")
    return model


if __name__ == "__main__":
    main()
