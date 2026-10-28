"""
Catastrophic forgetting in small open-source LLMs during sequential fine-tuning:
full fine-tuning (FFT) vs LoRA.  Assignment 4 experiment runner.

One "run" = one (model, task order, method, seed).  A run trains the model on three
tasks one after another and evaluates old knowledge (ARC-Challenge + MMLU) after
every step.  Each finished run is written to <out>/runs/<run_id>.json, and runs that
already exist are skipped, so a crashed or disconnected session can simply be
restarted with the same command.

Examples
    python forgetting.py --model Qwen/Qwen2.5-0.5B --seeds 42 43 44
    python forgetting.py --model Qwen/Qwen2.5-0.5B --seeds 42 --order reversed
    python forgetting.py --model Qwen/Qwen2.5-1.5B --seeds 42 43 44
    python forgetting.py --model Qwen/Qwen2.5-0.5B --seeds 42 --quick   # 3-minute smoke test
"""
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import transformers
import peft
import datasets
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, set_seed)

# ----------------------------------------------------------------------------------
# Frozen protocol (Appendix A).  Do not change these values after the freeze upload.
# ----------------------------------------------------------------------------------
DATA_SEED = 42                     # fixes training subsets and evaluation items for ALL runs
FROZEN = dict(
    n_train=1000,                  # training examples per sequential step
    n_heldout=100,                 # held-out examples per task (learning gain ΔL)
    n_arc=500,                     # ARC-Challenge test items (old knowledge)
    n_mmlu=500,                    # MMLU test items (old knowledge)
    epochs=2,
    max_len=384,
    effective_batch=32,
    lr={"FFT": 2e-5, "LoRA": 2e-4},
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    lora=dict(r=16, alpha=32, dropout=0.05,
              modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]),
)
QUICK = dict(n_train=64, n_heldout=16, n_arc=50, n_mmlu=50, epochs=1)

ORDERS = {
    "main":     ["GSM8K", "CodeAlpaca", "SciQ"],
    "reversed": ["SciQ", "CodeAlpaca", "GSM8K"],
}
LETTERS = ["A", "B", "C", "D", "E"]
PROMPT = "### Instruction:\n{inst}\n\n### Response:\n"


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def short_name(model_id):
    return model_id.split("/")[-1].replace("Qwen2.5-", "qwen").lower()   # qwen0.5b, qwen1.5b


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment():
    cuda = torch.cuda.is_available()
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0,
    }
    return env


def precision():
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8   # A100/A40/L40S/A6000 -> bf16
    fp16 = cuda and not bf16                                       # T4 -> fp16
    return bf16, fp16


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def fmt_gsm8k(x):
    return {"inst": f"Solve the problem step by step.\n{x['question']}", "resp": x["answer"]}


def fmt_code(x):
    inst = x["instruction"] + (f"\nInput: {x['input']}" if x.get("input") else "")
    return {"inst": inst, "resp": x["output"]}


def fmt_sciq(x):
    context = " ".join(x["support"].split()[:150])
    return {"inst": f"Answer the science question.\nContext: {context}\nQuestion: {x['question']}",
            "resp": x["correct_answer"]}


TASK_SPECS = {
    "GSM8K":      ("openai/gsm8k", "main", fmt_gsm8k),
    "CodeAlpaca": ("sahil2801/CodeAlpaca-20k", None, fmt_code),
    "SciQ":       ("allenai/sciq", None, fmt_sciq),
}


def make_encoder(tok, max_len):
    def encode(example):
        prompt = PROMPT.format(inst=example["inst"])
        full = prompt + example["resp"] + tok.eos_token
        kw = dict(truncation=True, max_length=max_len, add_special_tokens=False)
        p_len = len(tok(prompt, **kw)["input_ids"])
        enc = tok(full, **kw)
        # loss only on the response tokens
        enc["labels"] = [-100] * min(p_len, len(enc["input_ids"])) + enc["input_ids"][p_len:]
        return enc
    return encode


def load_task(name, tok, cfg):
    hf_id, config, fmt = TASK_SPECS[name]
    raw = load_dataset(hf_id, config, split="train") if config else load_dataset(hf_id, split="train")
    need = cfg["n_train"] + cfg["n_heldout"]
    raw = raw.shuffle(seed=DATA_SEED).select(range(min(len(raw), need * 2)))
    raw = raw.map(fmt, remove_columns=raw.column_names)
    raw = raw.filter(lambda x: len(x["resp"].strip()) > 0)
    enc = raw.map(make_encoder(tok, cfg["max_len"]), remove_columns=["inst", "resp"])
    enc = enc.filter(lambda e: any(l != -100 for l in e["labels"]))
    assert len(enc) >= need, f"{name}: not enough usable examples"
    return {"name": name,
            "train": enc.select(range(cfg["n_train"])),
            "heldout": enc.select(range(cfg["n_train"], need))}


def mc_prompt(question, choices):
    opts = "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(choices))
    return f"Question: {question.strip()}\n{opts}Answer:"


def load_old_items(cfg):
    items = []
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=DATA_SEED)
    n = 0
    for x in arc:
        if n == cfg["n_arc"]:
            break
        labels = x["choices"]["label"]
        if x["answerKey"] not in labels or len(labels) > 5:
            continue
        items.append({"id": f"arc::{x['id']}", "src": "ARC-Challenge",
                      "prompt": mc_prompt(x["question"], x["choices"]["text"]),
                      "n_opt": len(labels), "gold": labels.index(x["answerKey"])})
        n += 1
    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=DATA_SEED).select(range(cfg["n_mmlu"]))
    for i, x in enumerate(mmlu):
        items.append({"id": f"mmlu::{x['subject']}::{i}", "src": "MMLU",
                      "prompt": mc_prompt(x["question"], x["choices"]),
                      "n_opt": len(x["choices"]), "gold": int(x["answer"])})
    return items


def items_hash(items):
    payload = json.dumps([(it["id"], it["prompt"], it["gold"]) for it in items], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------
@torch.no_grad()
def eval_old_knowledge(model, tok, items, letter_ids, batch_size=16):
    """Log-likelihood scoring of the answer letter.

    For every item we keep: predicted option, probability of each option (renormalised
    over the valid letters) and 'letter mass' = total probability the model puts on the
    valid letters out of the whole vocabulary.  A falling letter mass means the model no
    longer wants to answer with a letter (format drift) rather than picking a wrong one.
    """
    model.eval()
    device = next(model.parameters()).device
    pad_side = tok.padding_side
    tok.padding_side = "left"
    pred, correct, mass, probs = [], [], [], []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        enc = tok([it["prompt"] for it in batch], return_tensors="pt", padding=True).to(device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        # fp32 on every GPU (comparable across T4 / A40); only the last position goes
        # through the LM head, otherwise logits for all tokens would need several GB
        hidden = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             position_ids=pos).last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden).float()
        full = torch.softmax(logits, dim=-1)
        for row, it in zip(full, batch):
            p = row[letter_ids[:it["n_opt"]]]
            m = float(p.sum())
            pn = (p / p.sum()).tolist()
            k = int(np.argmax(pn))
            pred.append(k)
            correct.append(int(k == it["gold"]))
            mass.append(round(m, 6))
            probs.append([round(v, 5) for v in pn])
    tok.padding_side = pad_side
    return {"pred": pred, "correct": correct, "letter_mass": mass, "probs": probs}


def acc_summary(ev, src):
    c = np.array(ev["correct"])
    return {"old_acc": round(float(100 * c.mean()), 3),
            "ARC": round(float(100 * c[src == "ARC-Challenge"].mean()), 3),
            "MMLU": round(float(100 * c[src == "MMLU"].mean()), 3),
            "mean_letter_mass": round(float(np.mean(ev["letter_mass"])), 5)}


@torch.no_grad()
def heldout_loss(model, ds):
    model.eval()
    device = next(model.parameters()).device
    losses = []
    for ex in ds:
        ids = torch.tensor([ex["input_ids"]], device=device)
        lab = torch.tensor([ex["labels"]], device=device)
        losses.append(model(input_ids=ids, labels=lab).loss.float().item())
    return float(np.mean(losses))


# ----------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------
def load_base_model(model_id):
    # weights are kept in fp32 (master copy); speed comes from fp16/bf16 autocast
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.float().to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False
    return model


def n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def per_device_batch():
    """Same effective batch (32) everywhere; only the split between batch and
    accumulation changes with GPU memory, which affects speed, not the update."""
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9:
        return 8, 4
    return 2, 16


def train_one_step(model, method, task, tok, cfg, seed, workdir):
    bs, acc = per_device_batch()
    steps = math.ceil(len(task["train"]) / (bs * acc)) * cfg["epochs"]
    bf16, fp16 = precision()
    args = TrainingArguments(
        output_dir=os.path.join(workdir, f"ckpt_{method}_{task['name']}"),
        per_device_train_batch_size=bs, gradient_accumulation_steps=acc,
        learning_rate=cfg["lr"][method], num_train_epochs=cfg["epochs"],
        lr_scheduler_type="cosine", warmup_steps=max(1, int(cfg["warmup_ratio"] * steps)),
        max_grad_norm=cfg["max_grad_norm"],
        gradient_checkpointing=(method == "FFT"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=4, save_strategy="no", report_to="none",
        fp16=fp16, bf16=bf16, seed=seed, data_seed=seed, disable_tqdm=True)
    collator = DataCollatorForSeq2Seq(tok, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=task["train"], data_collator=collator)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    curve = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    del trainer
    return out.training_loss, curve, dt, peak


def run_one(model_id, order, method, seed, tasks, items, src, base_eval, cfg, out_dir, workdir):
    run_id = f"{short_name(model_id)}_{order}_{method}_s{seed}"
    path = os.path.join(out_dir, "runs", f"{run_id}.json")
    if os.path.exists(path):
        log(f"skip {run_id} (already done)")
        return
    log(f"=== start {run_id} ===")
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = letter_token_ids(tok)
    model = load_base_model(model_id)
    total = n_params(model)
    record = {
        "run_id": run_id, "model": model_id, "order": order, "method": method, "seed": seed,
        "config": {k: v for k, v in cfg.items()}, "data_seed": DATA_SEED,
        "per_device_batch_x_accum": list(per_device_batch()),
        "precision": "bf16 mixed" if precision()[0] else ("fp16 mixed" if precision()[1] else "fp32"),
        "environment": environment(), "eval_items_sha256": items_hash(items),
        "total_params": total, "steps": [], "eval": {"0": base_eval},
    }
    record["steps"].append({"step": 0, "task": "baseline", **acc_summary(base_eval, src)})
    t_run = time.time()
    for t, task in enumerate(tasks, start=1):
        loss_before = heldout_loss(model, task["heldout"])
        if method == "LoRA":
            L = cfg["lora"]
            model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=L["r"],
                                                     lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                                                     target_modules=L["modules"]))
        trainable = n_params(model, trainable_only=True)
        train_loss, curve, dt, peak = train_one_step(model, method, task, tok, cfg, seed, workdir)
        free_gpu()                                 # drop optimizer state before evaluation
        if method == "LoRA":
            model = model.merge_and_unload()       # updates accumulate like in FFT
        loss_after = heldout_loss(model, task["heldout"])
        ev = eval_old_knowledge(model, tok, items, letter_ids)
        s = acc_summary(ev, src)
        record["eval"][str(t)] = ev
        record["steps"].append({
            "step": t, "task": task["name"], **s,
            "forgetting_pp": round(record["steps"][0]["old_acc"] - s["old_acc"], 3),
            "loss_before": round(loss_before, 5), "loss_after": round(loss_after, 5),
            "learning_gain": round(loss_before - loss_after, 5),
            "train_loss": round(train_loss, 5), "train_time_s": round(dt, 1),
            "peak_vram_gb": round(peak, 2), "trainable_params": trainable,
            "loss_curve": curve})
        log(f"[{run_id}] step {t} {task['name']}: old acc {s['old_acc']:.2f}% "
            f"(ΔF {record['steps'][-1]['forgetting_pp']:+.2f} pp) | held-out loss "
            f"{loss_before:.3f} -> {loss_after:.3f} | {dt:.0f}s | peak {peak:.1f} GB")
    record["run_time_s"] = round(time.time() - t_run, 1)
    del model
    free_gpu()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, path)
    log(f"=== saved {path} ===")


def letter_token_ids(tok):
    ids = [tok(" " + L, add_special_tokens=False)["input_ids"] for L in LETTERS]
    assert all(len(i) == 1 for i in ids), "answer letters must be single tokens"
    return [i[0] for i in ids]


def baseline_eval(model_id, items, out_dir):
    """Step-0 evaluation of the untouched model, computed once per model and cached."""
    path = os.path.join(out_dir, "baselines", f"{short_name(model_id)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["eval"]
    log(f"baseline evaluation of {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_base_model(model_id)
    ev = eval_old_knowledge(model, tok, items, letter_token_ids(tok))
    del model
    free_gpu()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"model": model_id, "eval_items_sha256": items_hash(items), "eval": ev}, f)
    return ev


def save_environment(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=2)
    try:
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout
        with open(os.path.join(out_dir, "pip_freeze.txt"), "w") as f:
            f.write(frozen)
    except Exception as e:                                   # never block a run on this
        log(f"pip freeze failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--order", choices=list(ORDERS), default="main")
    ap.add_argument("--methods", nargs="+", default=["FFT", "LoRA"], choices=["FFT", "LoRA"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--workdir", default="/tmp/rtm_ckpt")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test, written to <out>_quick")
    ap.add_argument("--items-only", action="store_true",
                    help="only build the 1,000 evaluation items, print their SHA-256 and exit (for Appendix A)")
    a = ap.parse_args()

    cfg = dict(FROZEN)
    out_dir = a.out
    if a.quick:
        cfg.update(QUICK)
        out_dir = a.out.rstrip("/") + "_quick"
    os.makedirs(out_dir, exist_ok=True)

    items = load_old_items(cfg)
    src = np.array([it["src"] for it in items])
    with open(os.path.join(out_dir, "eval_items.json"), "w") as f:
        json.dump({"sha256": items_hash(items), "items": items}, f, ensure_ascii=False)
    log(f"old-knowledge items: {len(items)} "
        f"(ARC {int((src == 'ARC-Challenge').sum())}, MMLU {int((src == 'MMLU').sum())})")
    log(f"eval_items sha256 = {items_hash(items)}")
    if a.items_only:
        return

    save_environment(out_dir)
    log(f"model={a.model} order={a.order} seeds={a.seeds} methods={a.methods} out={out_dir}")
    log(f"environment: {environment()}")
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tasks = [load_task(name, tok, cfg) for name in ORDERS[a.order]]
    for t in tasks:
        log(f"task {t['name']}: train={len(t['train'])} held-out={len(t['heldout'])}")

    base = baseline_eval(a.model, items, out_dir)
    log(f"baseline: {acc_summary(base, src)}")
    for seed in a.seeds:
        for method in a.methods:
            run_one(a.model, a.order, method, seed, tasks, items, src, base, cfg, out_dir, a.workdir)
    log("all requested runs finished")


if __name__ == "__main__":
    main()
