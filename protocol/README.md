# Protocol check: is the FFT–LoRA gap an artefact of the evaluation?

Follow-up to Assignment 4. Question: does the 10.8 pp gap between FFT and LoRA forgetting
(Qwen2.5-0.5B, Step 3, main order) change when the answers are scored differently or the
options are shuffled?

| Mode | Prompt / options | Prediction |
|---|---|---|
| `letter_orig` | original prompt and option order | letter with the highest next-token probability (the Assignment 4 metric, same code) |
| `letter_shuf` | 5 fixed shuffles per item (`shuffles.json`, seed 20261116; distinct, never the original order); gold letter remapped | same letter scoring; accuracy = mean over the 5 shuffles |
| `full_answer` | original prompt and option order, then `" <option text>"` after `Answer:` | option with the highest mean log-probability per token |

Forgetting = Step 0 accuracy − Step 3 accuracy. Gap = FFT forgetting − LoRA forgetting.
Step 0 is the same base model for both methods, so the gap equals LoRA Step-3 accuracy − FFT
Step-3 accuracy. Gap change = gap(mode) − gap(`letter_orig`). 95% CIs: paired bootstrap over
items (2,000 resamples).

**Checkpoints.** Assignment 4 did not save weights, so Step 3 is re-trained with the frozen
functions of `forgetting.py` (unchanged). The original letter score of every re-trained checkpoint
is compared item by item with the Assignment 4 logs (`repro/*.json`, `reproduction.csv`).
All three modes are computed on the *same* re-trained checkpoints, so the mode comparison does
not depend on how exactly the re-training reproduces the logs.

## Reproduce (Colab T4, ≈ 2.5 h)

```bash
pip install transformers==5.21.0 peft==0.21.2 datasets==5.1.0 accelerate==1.15.0 tabulate
python protocol/protocol_eval.py --step0-only --out results_protocol      # Day 1: Step 0 must match the log exactly
python protocol/protocol_eval.py --seeds 42 43 44 --out results_protocol  # re-train Step 3 + 3 modes (resumable)
python protocol/protocol_analysis.py --res results_protocol --out report_protocol
python protocol/protocol_analysis.py --res results_protocol --verify letter_shuf FFT 42 All
```

On a non-T4 GPU (e.g. RunPod) add `--match-t4` so Step 3 is trained with the Assignment 4 recipe (fp16 mixed, batch 2 × 16). Or open `run_protocol_colab.ipynb`. Add `--save-weights DIR` to keep the Step-3 weights and
`--load-weights DIR` to re-score saved weights without training.

## Outputs

| File | Content |
|---|---|
| `results_protocol/items/*.csv` | item-level rows: checkpoint, mode, shuffle, permutation, gold and predicted position, correct, letter mass, scores |
| `results_protocol/shuffles.json` | the 5 permutations per item used for every checkpoint |
| `results_protocol/repro/*.json` | reproduction of the Assignment 4 letter scores (Step 0, Steps 1–3) |
| `report_protocol/table_protocol.csv` | Step 0 / Step 3 accuracy and forgetting per mode × subset (ARC, MMLU, All) × seed × method |
| `report_protocol/gap_per_seed.csv`, `gap_summary.csv` | gap per seed with CI, change vs `letter_orig` |
| `report_protocol/position_shuffled.csv` | shuffled mode: accuracy by position of the correct answer, share of each predicted letter |
| `report_protocol/fig_gap_by_mode.png` | gap per mode, seeds visible |
| `report_protocol/tables.md` | all tables in one file |
