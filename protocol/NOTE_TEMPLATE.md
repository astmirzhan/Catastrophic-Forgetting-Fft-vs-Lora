# Research note: sensitivity of the FFT–LoRA forgetting gap to the evaluation protocol

*Myssyrgaliyev M., Muratbaiuly A. — one page. Fill every [ ] from report_protocol/; do not round differently from the tables.*

**Question.** Assignment 4 found that, after GSM8K → CodeAlpaca → SciQ, Qwen2.5-0.5B lost 10.8 pp
more ARC/MMLU accuracy under FFT than under LoRA (letter scoring, mean of 3 seeds). The error
analysis suggested that most FFT forgetting was lost answer format plus an "A" bias. Does the gap
survive (a) shuffled options and (b) scoring the answer text instead of the letter?

**What we did.** Same 1,000 items (SHA-256 e6e4c60c…), Step 0 = base model, Step 3 re-trained
with the frozen code because Assignment 4 saved no weights. Coverage: seeds [ ].
Re-trained Step-3 letter accuracy differed from the logs by [ ] pp on average
(max [ ] pp; Step 0 reproduced exactly: [ ]). Three modes: original letter, 5 fixed shuffles
(gold remapped, identical permutations for all checkpoints), full-answer mean log-prob per token.

**What changed.**

| Mode | FFT forgetting | LoRA forgetting | Gap (mean ± SD) | Per seed | Change vs letter |
|---|---|---|---|---|---|
| Letter, original | [ ] | [ ] | [ ] | [ ] | — |
| Letter, 5 shuffles | [ ] | [ ] | [ ] | [ ] | [ ] |
| Full-answer | [ ] | [ ] | [ ] | [ ] | [ ] |

When scoring changed from letters to full answers, the measured gap [narrowed / widened /
stayed similar] by [ ] pp ([ ] → [ ] pp). Shuffling options changed it by [ ] pp.
ARC vs MMLU: [ ].

**Position effect.** With shuffled options, FFT Step-3 accuracy was [ ]% when the correct answer was
A and [ ]% when it was D; FFT predicted A for [ ]% of questions (base model [ ]%). [Shuffling spreads
the A-bias across positions, so it costs/helps …]

**Consistency across seeds.** [All / 2 of 3 / no] seeds moved in the same direction; the gap change
had a bootstrap 95% CI excluding 0 in [ ] of [ ] seeds.

**What the evidence supports.** [If narrows:] The letter-scoring gap was sensitive to the protocol:
[ ]% of it ([ ] of 10.8 pp) disappears under full-answer scoring, so that part reflects lost
answer format rather than lost knowledge. A residual gap of [ ] pp remains [significant / not].
[If persists:] The gap survived both option shuffling and answer-text scoring, so it is not an
artefact of letter format or answer position.

**Limits.** n = [ ] seeds; re-trained checkpoints are not bit-identical to Assignment 4;
full-answer scoring favours short/frequent answer strings (mean per token reduces but does not
remove this); one model size and one task order.
