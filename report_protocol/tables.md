Matched seeds: [42, 43, 44]

## Results table

| mode        | subset   |   seed | method   |   n_items |   step0_acc |   step3_acc |   forgetting_pp |   step3_acc_shuffle_min |   step3_acc_shuffle_max |
|:------------|:---------|-------:|:---------|----------:|------------:|------------:|----------------:|------------------------:|------------------------:|
| letter_orig | ARC      |     42 | FFT      |       500 |       56.2  |       35    |           21.2  |                   nan   |                   nan   |
| letter_orig | ARC      |     42 | LoRA     |       500 |       56.2  |       55.8  |            0.4  |                   nan   |                   nan   |
| letter_orig | ARC      |     43 | FFT      |       500 |       56.2  |       44.4  |           11.8  |                   nan   |                   nan   |
| letter_orig | ARC      |     43 | LoRA     |       500 |       56.2  |       54.4  |            1.8  |                   nan   |                   nan   |
| letter_orig | ARC      |     44 | FFT      |       500 |       56.2  |       45.2  |           11    |                   nan   |                   nan   |
| letter_orig | ARC      |     44 | LoRA     |       500 |       56.2  |       51.2  |            5    |                   nan   |                   nan   |
| letter_orig | MMLU     |     42 | FFT      |       500 |       47.6  |       31.8  |           15.8  |                   nan   |                   nan   |
| letter_orig | MMLU     |     42 | LoRA     |       500 |       47.6  |       48.2  |           -0.6  |                   nan   |                   nan   |
| letter_orig | MMLU     |     43 | FFT      |       500 |       47.6  |       44.2  |            3.4  |                   nan   |                   nan   |
| letter_orig | MMLU     |     43 | LoRA     |       500 |       47.6  |       46.2  |            1.4  |                   nan   |                   nan   |
| letter_orig | MMLU     |     44 | FFT      |       500 |       47.6  |       41.8  |            5.8  |                   nan   |                   nan   |
| letter_orig | MMLU     |     44 | LoRA     |       500 |       47.6  |       49.2  |           -1.6  |                   nan   |                   nan   |
| letter_orig | All      |     42 | FFT      |      1000 |       51.9  |       33.4  |           18.5  |                   nan   |                   nan   |
| letter_orig | All      |     42 | LoRA     |      1000 |       51.9  |       52    |           -0.1  |                   nan   |                   nan   |
| letter_orig | All      |     43 | FFT      |      1000 |       51.9  |       44.3  |            7.6  |                   nan   |                   nan   |
| letter_orig | All      |     43 | LoRA     |      1000 |       51.9  |       50.3  |            1.6  |                   nan   |                   nan   |
| letter_orig | All      |     44 | FFT      |      1000 |       51.9  |       43.5  |            8.4  |                   nan   |                   nan   |
| letter_orig | All      |     44 | LoRA     |      1000 |       51.9  |       50.2  |            1.7  |                   nan   |                   nan   |
| letter_shuf | ARC      |     42 | FFT      |       500 |       55.64 |       33.28 |           22.36 |                    32.4 |                    34.6 |
| letter_shuf | ARC      |     42 | LoRA     |       500 |       55.64 |       55.08 |            0.56 |                    54   |                    56.4 |
| letter_shuf | ARC      |     43 | FFT      |       500 |       55.64 |       45.72 |            9.92 |                    43.4 |                    47.4 |
| letter_shuf | ARC      |     43 | LoRA     |       500 |       55.64 |       54.92 |            0.72 |                    53.4 |                    57.2 |
| letter_shuf | ARC      |     44 | FFT      |       500 |       55.64 |       44.68 |           10.96 |                    42.6 |                    45.8 |
| letter_shuf | ARC      |     44 | LoRA     |       500 |       55.64 |       52.28 |            3.36 |                    50.6 |                    54.2 |
| letter_shuf | MMLU     |     42 | FFT      |       500 |       46.4  |       30.2  |           16.2  |                    27.2 |                    32.4 |
| letter_shuf | MMLU     |     42 | LoRA     |       500 |       46.4  |       46.84 |           -0.44 |                    44.8 |                    49.2 |
| letter_shuf | MMLU     |     43 | FFT      |       500 |       46.4  |       37.64 |            8.76 |                    36.4 |                    38.2 |
| letter_shuf | MMLU     |     43 | LoRA     |       500 |       46.4  |       46.8  |           -0.4  |                    44.6 |                    49.6 |
| letter_shuf | MMLU     |     44 | FFT      |       500 |       46.4  |       36.84 |            9.56 |                    35.8 |                    38.2 |
| letter_shuf | MMLU     |     44 | LoRA     |       500 |       46.4  |       44.44 |            1.96 |                    42.8 |                    46.8 |
| letter_shuf | All      |     42 | FFT      |      1000 |       51.02 |       31.74 |           19.28 |                    29.8 |                    32.6 |
| letter_shuf | All      |     42 | LoRA     |      1000 |       51.02 |       50.96 |            0.06 |                    49.6 |                    52.8 |
| letter_shuf | All      |     43 | FFT      |      1000 |       51.02 |       41.68 |            9.34 |                    40.6 |                    42.7 |
| letter_shuf | All      |     43 | LoRA     |      1000 |       51.02 |       50.86 |            0.16 |                    49.7 |                    53.4 |
| letter_shuf | All      |     44 | FFT      |      1000 |       51.02 |       40.76 |           10.26 |                    39.2 |                    41.9 |
| letter_shuf | All      |     44 | LoRA     |      1000 |       51.02 |       48.36 |            2.66 |                    47.3 |                    50.5 |
| full_answer | ARC      |     42 | FFT      |       500 |       36    |       35.4  |            0.6  |                   nan   |                   nan   |
| full_answer | ARC      |     42 | LoRA     |       500 |       36    |       43.8  |           -7.8  |                   nan   |                   nan   |
| full_answer | ARC      |     43 | FFT      |       500 |       36    |       40.6  |           -4.6  |                   nan   |                   nan   |
| full_answer | ARC      |     43 | LoRA     |       500 |       36    |       39.2  |           -3.2  |                   nan   |                   nan   |
| full_answer | ARC      |     44 | FFT      |       500 |       36    |       35.8  |            0.2  |                   nan   |                   nan   |
| full_answer | ARC      |     44 | LoRA     |       500 |       36    |       43.4  |           -7.4  |                   nan   |                   nan   |
| full_answer | MMLU     |     42 | FFT      |       500 |       28.4  |       30.2  |           -1.8  |                   nan   |                   nan   |
| full_answer | MMLU     |     42 | LoRA     |       500 |       28.4  |       34    |           -5.6  |                   nan   |                   nan   |
| full_answer | MMLU     |     43 | FFT      |       500 |       28.4  |       32.6  |           -4.2  |                   nan   |                   nan   |
| full_answer | MMLU     |     43 | LoRA     |       500 |       28.4  |       31    |           -2.6  |                   nan   |                   nan   |
| full_answer | MMLU     |     44 | FFT      |       500 |       28.4  |       28.4  |            0    |                   nan   |                   nan   |
| full_answer | MMLU     |     44 | LoRA     |       500 |       28.4  |       33.6  |           -5.2  |                   nan   |                   nan   |
| full_answer | All      |     42 | FFT      |      1000 |       32.2  |       32.8  |           -0.6  |                   nan   |                   nan   |
| full_answer | All      |     42 | LoRA     |      1000 |       32.2  |       38.9  |           -6.7  |                   nan   |                   nan   |
| full_answer | All      |     43 | FFT      |      1000 |       32.2  |       36.6  |           -4.4  |                   nan   |                   nan   |
| full_answer | All      |     43 | LoRA     |      1000 |       32.2  |       35.1  |           -2.9  |                   nan   |                   nan   |
| full_answer | All      |     44 | FFT      |      1000 |       32.2  |       32.1  |            0.1  |                   nan   |                   nan   |
| full_answer | All      |     44 | LoRA     |      1000 |       32.2  |       38.5  |           -6.3  |                   nan   |                   nan   |

## Gap summary

| subset   | mode        |   n_seeds |   gap_mean |   gap_sd | gap_per_seed      |   change_mean | change_per_seed      |   seeds_with_ci_excluding_0 |
|:---------|:------------|----------:|-----------:|---------:|:------------------|--------------:|:---------------------|----------------------------:|
| ARC      | letter_orig |         3 |      12.27 |     7.66 | 20.8 / 10.0 / 6.0 |        nan    | nan                  |                         nan |
| ARC      | letter_shuf |         3 |      12.87 |     7.78 | 21.8 / 9.2 / 7.6  |          0.6  | +1.0 / -0.8 / +1.6   |                           0 |
| ARC      | full_answer |         3 |       4.87 |     5.44 | 8.4 / -1.4 / 7.6  |         -7.4  | -12.4 / -11.4 / +1.6 |                           2 |
| MMLU     | letter_orig |         3 |       8.6  |     7.27 | 16.4 / 2.0 / 7.4  |        nan    | nan                  |                         nan |
| MMLU     | letter_shuf |         3 |      11.13 |     4.83 | 16.6 / 9.2 / 7.6  |          2.53 | +0.2 / +7.2 / +0.2   |                           1 |
| MMLU     | full_answer |         3 |       2.47 |     3.59 | 3.8 / -1.6 / 5.2  |         -6.13 | -12.6 / -3.6 / -2.2  |                           1 |
| All      | letter_orig |         3 |      10.43 |     7.08 | 18.6 / 6.0 / 6.7  |        nan    | nan                  |                         nan |
| All      | letter_shuf |         3 |      12    |     6.3  | 19.2 / 9.2 / 7.6  |          1.57 | +0.6 / +3.2 / +0.9   |                           0 |
| All      | full_answer |         3 |       3.67 |     4.48 | 6.1 / -1.5 / 6.4  |         -6.77 | -12.5 / -7.5 / -0.3  |                           2 |

## Gap per seed (95% bootstrap CI)

| subset   |   seed | mode        |   gap_pp |   gap_ci_lo |   gap_ci_hi |   change_vs_orig_pp |   change_ci_lo |   change_ci_hi |
|:---------|-------:|:------------|---------:|------------:|------------:|--------------------:|---------------:|---------------:|
| ARC      |     42 | letter_orig |    20.8  |       15.6  |       26.4  |              nan    |         nan    |         nan    |
| ARC      |     42 | letter_shuf |    21.8  |       18.04 |       25.4  |                1    |          -4.48 |           5.8  |
| ARC      |     42 | full_answer |     8.4  |        4    |       12.8  |              -12.4  |         -18.8  |          -6.2  |
| ARC      |     43 | letter_orig |    10    |        5.2  |       14.8  |              nan    |         nan    |         nan    |
| ARC      |     43 | letter_shuf |     9.2  |        6.4  |       12    |               -0.8  |          -5.56 |           3.76 |
| ARC      |     43 | full_answer |    -1.4  |       -6.2  |        3.2  |              -11.4  |         -17.6  |          -5.6  |
| ARC      |     44 | letter_orig |     6    |        1.2  |       11.4  |              nan    |         nan    |         nan    |
| ARC      |     44 | letter_shuf |     7.6  |        4.76 |       10.36 |                1.6  |          -3.36 |           6.2  |
| ARC      |     44 | full_answer |     7.6  |        3.6  |       11.6  |                1.6  |          -4.6  |           7.6  |
| MMLU     |     42 | letter_orig |    16.4  |       11.2  |       21.6  |              nan    |         nan    |         nan    |
| MMLU     |     42 | letter_shuf |    16.64 |       13.32 |       20    |                0.24 |          -4.68 |           5.6  |
| MMLU     |     42 | full_answer |     3.8  |       -0.6  |        8    |              -12.6  |         -19.01 |          -5.8  |
| MMLU     |     43 | letter_orig |     2    |       -2.4  |        6.6  |              nan    |         nan    |         nan    |
| MMLU     |     43 | letter_shuf |     9.16 |        6.64 |       11.76 |                7.16 |           2.6  |          11.68 |
| MMLU     |     43 | full_answer |    -1.6  |       -5.8  |        2.6  |               -3.6  |          -9.6  |           2.2  |
| MMLU     |     44 | letter_orig |     7.4  |        2.4  |       12.2  |              nan    |         nan    |         nan    |
| MMLU     |     44 | letter_shuf |     7.6  |        4.84 |       10.52 |                0.2  |          -5    |           5.48 |
| MMLU     |     44 | full_answer |     5.2  |        1.2  |        9.4  |               -2.2  |          -8    |           3.8  |
| All      |     42 | letter_orig |    18.6  |       14.8  |       22.5  |              nan    |         nan    |         nan    |
| All      |     42 | letter_shuf |    19.22 |       16.66 |       21.78 |                0.62 |          -3    |           4.32 |
| All      |     42 | full_answer |     6.1  |        2.8  |        9.2  |              -12.5  |         -17    |          -8.1  |
| All      |     43 | letter_orig |     6    |        2.6  |        9.2  |              nan    |         nan    |         nan    |
| All      |     43 | letter_shuf |     9.18 |        7.18 |       11.16 |                3.18 |          -0.02 |           6.34 |
| All      |     43 | full_answer |    -1.5  |       -4.7  |        1.6  |               -7.5  |         -11.6  |          -3.4  |
| All      |     44 | letter_orig |     6.7  |        3.2  |       10.2  |              nan    |         nan    |         nan    |
| All      |     44 | letter_shuf |     7.6  |        5.68 |        9.74 |                0.9  |          -2.84 |           4.5  |
| All      |     44 | full_answer |     6.4  |        3.5  |        9.3  |               -0.3  |          -4.8  |           4.2  |

## Reproduction of Assignment 4 logs (letter_orig)

| ckpt                         |   acc_now |   acc_logged |   diff_pp |   same_prediction_pct |   max_abs_letter_mass_diff |   step1_diff_pp |   step2_diff_pp |
|:-----------------------------|----------:|-------------:|----------:|----------------------:|---------------------------:|----------------:|----------------:|
| qwen0.5b_main_FFT_s42_step3  |      33.4 |         33.4 |       0   |                  97.3 |                   0.10507  |            -0.2 |             0   |
| qwen0.5b_main_FFT_s43_step3  |      44.3 |         44.3 |       0   |                  98.8 |                   0.052154 |            -0.2 |             0.1 |
| qwen0.5b_main_FFT_s44_step3  |      43.5 |         43.6 |      -0.1 |                  99.2 |                   0.020899 |             0.1 |             0.3 |
| qwen0.5b_main_LoRA_s42_step3 |      52   |         51.7 |       0.3 |                  97.8 |                   0.079774 |             0.2 |            -0.4 |
| qwen0.5b_main_LoRA_s43_step3 |      50.3 |         51.9 |      -1.6 |                  96.9 |                   0.150665 |            -0.1 |             0   |
| qwen0.5b_main_LoRA_s44_step3 |      50.2 |         50.1 |       0.1 |                  96.9 |                   0.102966 |             0   |             0.3 |
| qwen0.5b_step0               |      51.9 |         51.9 |       0   |                 100   |                   1e-06    |           nan   |           nan   |

## Shuffled options: accuracy by position of the correct answer

| ckpt                         | method   |   seed | letter   |   acc_when_gold_here |   n_gold_here |   pred_share |
|:-----------------------------|:---------|-------:|:---------|---------------------:|--------------:|-------------:|
| qwen0.5b_step0               | base     |        | A        |                45.81 |          1253 |        20.32 |
| qwen0.5b_step0               | base     |        | B        |                51.29 |          1242 |        25.47 |
| qwen0.5b_step0               | base     |        | C        |                48.54 |          1304 |        23.05 |
| qwen0.5b_step0               | base     |        | D        |                58.52 |          1191 |        31.16 |
| qwen0.5b_main_FFT_s42_step3  | FFT      |     42 | A        |                69.11 |          1253 |        62.2  |
| qwen0.5b_main_FFT_s42_step3  | FFT      |     42 | B        |                22.62 |          1242 |        15.01 |
| qwen0.5b_main_FFT_s42_step3  | FFT      |     42 | C        |                19.02 |          1304 |        13.13 |
| qwen0.5b_main_FFT_s42_step3  | FFT      |     42 | D        |                15.62 |          1191 |         9.66 |
| qwen0.5b_main_LoRA_s42_step3 | LoRA     |     42 | A        |                44.77 |          1253 |        20.14 |
| qwen0.5b_main_LoRA_s42_step3 | LoRA     |     42 | B        |                47.1  |          1242 |        22.85 |
| qwen0.5b_main_LoRA_s42_step3 | LoRA     |     42 | C        |                54.68 |          1304 |        27.17 |
| qwen0.5b_main_LoRA_s42_step3 | LoRA     |     42 | D        |                57.01 |          1191 |        29.84 |
| qwen0.5b_main_FFT_s43_step3  | FFT      |     43 | A        |                65.92 |          1253 |        45.79 |
| qwen0.5b_main_FFT_s43_step3  | FFT      |     43 | B        |                35.1  |          1242 |        19.52 |
| qwen0.5b_main_FFT_s43_step3  | FFT      |     43 | C        |                33.28 |          1304 |        17.84 |
| qwen0.5b_main_FFT_s43_step3  | FFT      |     43 | D        |                31.91 |          1191 |        16.85 |
| qwen0.5b_main_LoRA_s43_step3 | LoRA     |     43 | A        |                44.77 |          1253 |        19.48 |
| qwen0.5b_main_LoRA_s43_step3 | LoRA     |     43 | B        |                47.1  |          1242 |        22.44 |
| qwen0.5b_main_LoRA_s43_step3 | LoRA     |     43 | C        |                50.84 |          1304 |        25.33 |
| qwen0.5b_main_LoRA_s43_step3 | LoRA     |     43 | D        |                60.79 |          1191 |        32.75 |
| qwen0.5b_main_FFT_s44_step3  | FFT      |     44 | A        |                56.34 |          1253 |        37.78 |
| qwen0.5b_main_FFT_s44_step3  | FFT      |     44 | B        |                43.4  |          1242 |        26.63 |
| qwen0.5b_main_FFT_s44_step3  | FFT      |     44 | C        |                36.04 |          1304 |        22    |
| qwen0.5b_main_FFT_s44_step3  | FFT      |     44 | D        |                26.53 |          1191 |        13.59 |
| qwen0.5b_main_LoRA_s44_step3 | LoRA     |     44 | A        |                39.51 |          1253 |        17.45 |
| qwen0.5b_main_LoRA_s44_step3 | LoRA     |     44 | B        |                41.95 |          1242 |        19.76 |
| qwen0.5b_main_LoRA_s44_step3 | LoRA     |     44 | C        |                46.17 |          1304 |        22.63 |
| qwen0.5b_main_LoRA_s44_step3 | LoRA     |     44 | D        |                66.33 |          1191 |        40.16 |