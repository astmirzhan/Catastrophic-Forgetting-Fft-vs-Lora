# Catastrophic Forgetting in Small Open-Source LLMs During Sequential Fine-Tuning: Full Fine-Tuning vs LoRA

Research Methods, Assignment 4 (experiment execution). Builds on the Assignment 3 pilot.

**Research question.** How does the degree of catastrophic forgetting differ between full
fine-tuning (FFT) and LoRA when a small open-source LLM (≤1.5B parameters) is fine-tuned
sequentially on several new task datasets?

**Setup in one paragraph.** A Qwen2.5 base model is trained on three tasks one after another
(GSM8K → CodeAlpaca → SciQ, 1,000 examples each, 2 epochs). After every step we measure
*old knowledge* on a fixed set of 500 ARC-Challenge + 500 MMLU questions (answer-letter
log-likelihood) and *new learning* as the drop of held-out loss on the current task. Forgetting
ΔF = baseline accuracy − current accuracy (percentage points). LoRA (r=16, α=32, all linear
layers) is merged into the weights after each step, so updates accumulate as in FFT.
The protocol was frozen before the final runs (Appendix A in the report).

| Run set | Model | Order | Seeds | Where it was run |
|---|---|---|---|---|
| Main | Qwen2.5-0.5B | GSM8K → CodeAlpaca → SciQ | 42, 43, 44 | Colab T4 (fp16 mixed) |
| Order check | Qwen2.5-0.5B | SciQ → CodeAlpaca → GSM8K | 42 | Colab T4 |
| Scale check | Qwen2.5-1.5B | GSM8K → CodeAlpaca → SciQ | 42, 43, 44 | RunPod 48 GB GPU (bf16 mixed) |

## Repository layout

```
forgetting.py      experiment runner (one JSON per finished run, resumable)
analysis.py        tables, statistics, verdicts, error analysis, figures
run_colab.ipynb    the same commands as a Colab notebook (saves to Google Drive)
run_runpod.sh      1.5B scale check on a rented 48 GB GPU (quick | full)
requirements.txt   minimum package versions
results_*/         raw outputs: runs/*.json, baselines/, eval_items.json, environment.json, pip_freeze.txt, logs
report/            everything produced by analysis.py (CSV tables, summary.md, fig1–fig6)
```

## Reproduce

```bash
git clone https://github.com/<your-username>/catastrophic-forgetting-fft-vs-lora.git && cd catastrophic-forgetting-fft-vs-lora
pip install -r requirements.txt
pip uninstall -y torchao          # only if peft fails to import because of torchao (seen on Colab)

# 0a) build the 1,000 evaluation questions and print their SHA-256 (goes into Appendix A)
python forgetting.py --items-only --out results_colab

# 0b) smoke test, ~5 min on a T4; writes to results_quick/
python forgetting.py --model Qwen/Qwen2.5-0.5B --seeds 42 --quick

mkdir -p results_colab results_runpod

# 1) main experiment, 0.5B, 3 seeds × {FFT, LoRA}
python forgetting.py --model Qwen/Qwen2.5-0.5B --seeds 42 43 44 --out results_colab 2>&1 | tee -a results_colab/log_0.5b_main.txt

# 2) task-order check, 0.5B, 1 seed
python forgetting.py --model Qwen/Qwen2.5-0.5B --seeds 42 --order reversed --out results_colab 2>&1 | tee -a results_colab/log_0.5b_reversed.txt

# 3) scale check, 1.5B, 3 seeds (needs a ≥40 GB GPU for FFT: A40 / RTX A6000 / L40S / A100)
python forgetting.py --model Qwen/Qwen2.5-1.5B --seeds 42 43 44 --out results_runpod 2>&1 | tee -a results_runpod/log_1.5b_main.txt

# 4) analysis (merges all result folders)
python analysis.py --results results_colab results_runpod --out report
```

Every finished run is saved immediately as `results_*/runs/<model>_<order>_<method>_s<seed>.json`.
If the session dies, run the same command again: finished runs are skipped.

### Determinism and what is fixed

* `DATA_SEED = 42` fixes the training subsets, held-out subsets and the 1,000 evaluation
  questions for **all** runs. The SHA-256 of the evaluation set is stored in
  `eval_items.json` and inside every run file; `analysis.py` warns if they differ.
* The run seed (42/43/44) changes LoRA initialisation, data shuffling and dropout.
* Effective batch size is always 32. On GPUs with >30 GB it is 8×4 instead of 2×16
  (same update, faster). Mixed precision is fp16 on T4 and bf16 on Ampere/Ada GPUs; the
  weights themselves are fp32 in both cases. GPU-level nondeterminism means a re-run can
  differ by a few items; the seed-to-seed spread reported in the tables is much larger.

## What `analysis.py` reports

* `table_main.csv` — accuracy, ΔF, relative forgetting rate, forgotten/gained items, learning
  gain, letter mass; mean ± SD across seeds.
* `tests_per_seed.csv` — exact McNemar tests (baseline vs step; LoRA vs FFT), paired bootstrap
  95% CIs (2,000 resamples), Holm correction within each family of 3 steps.
* `tests_across_seeds.csv` — paired t-test, Wilcoxon, Cohen's d_z on ΔF(FFT) − ΔF(LoRA).
* `correlation.csv` — Spearman ρ between learning gain and forgetting (H3).
* `verdicts.csv` — H1–H3 decided by the rules frozen in Appendix A.
* `error_counts.csv`, `error_sample_FFT.csv`, `error_sample_LoRA.csv` — error analysis of
  forgotten items (0.5B, seed 42, after step 3) with automatic categories (format drift,
  letter bias, near tie, factual erasure) and an empty `manual_category` column for checking.
* `diagnostics.csv` — letter mass and answer-letter distribution per checkpoint.
* `efficiency.csv` — training time per step, peak GPU memory, trainable parameters.
* `fig1_trajectory.png` … `fig6_order.png`.

## License

MIT (see `LICENSE`). Datasets and models keep their own licenses (GSM8K MIT, CodeAlpaca
Apache-2.0 / CC-BY-NC for generated outputs, SciQ CC-BY-NC-3.0, ARC CC-BY-SA-4.0, MMLU MIT,
Qwen2.5 Apache-2.0).
