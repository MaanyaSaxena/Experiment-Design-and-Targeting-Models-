# Decision memo: Gate move, level 30 → 40
All numbers are taken from the final executed notebook.

**Question:** Does moving the first gate from level 30 to level 40 change player retention?

**Recommendation: do not ship the level-40 gate as tested.** The 7-day retention drop survives multiple-testing correction and the whole 95% interval is below zero, so there is no sign of a retention gain.

**Design:** randomised two-arm test, 90,189 players (gate_30 = 44,700; gate_40 = 45,489).
- Primary metric: 7-day retention. Secondary: 1-day retention and game rounds.
- Baseline assumed at 19% (the observed control rate was 19.02%, but the baseline was assumed, not taken from a prior period).
- Planned MDE: 1.0 pp absolute, about a 5% relative change. This is an assumption: it is the smallest 7-day retention shift I would treat as commercially meaningful for a free-to-play game, and the dataset has no revenue data to price it.
- Planned sample: 24,638 per group at alpha 0.05, power 0.80. Actual samples were larger: power to detect the planned 1 pp effect was about 96.5%, and the smallest effect detectable with 80% power was about 0.74 pp.

**Data checks**
- Duplicates 0; missing values 0.
- Sample-ratio check: chi-square = 6.90, p = 0.0086 (control 49.56% versus 50% expected). This is below 0.01 but above the stricter 0.001 alert level many teams use, so I treat it as a flag, not proof of broken randomisation. Next steps I would take: confirm how users were assigned, check whether logging dropped users differently by arm or by day, and see whether the imbalance sits in one date range or segment, then rerun excluding it to test whether the conclusion holds. The dataset has no assignment, platform or date fields, so these checks were not possible here.
- One gate_30 player has 49,854 rounds (next highest 2,961). The effect of this player on the game-rounds mean difference was not tested; the median and rank-based tests are less sensitive to it.

**Results**
| Metric | gate_30 | gate_40 | Difference (95% CI) | Raw p | Holm p |
|---|---|---|---|---|---|
| 1-day retention | 44.82% | 44.23% | -0.59 pp (-1.24 to +0.06) | 0.074 | 0.149 |
| 7-day retention | 19.02% | 18.20% | -0.82 pp (-1.33 to -0.31) | 0.0016 | 0.0047 |
| Game rounds, mean | n/a | n/a | -1.16 (permutation test) | 0.46 | 0.46 |
| Game rounds, median | n/a | n/a | -1 (permutation test) | 0.89 | n/a |

Bootstrap CI for 7-day retention: -1.33 to -0.32 pp, in line with the z-interval. Relative change in 7-day retention: about -4.3%. Mann-Whitney on rounds: p = 0.050, borderline; it tests a shift in the distribution, not the mean.

**Peeking risk (A/A simulation, 10 looks, 2,000 runs):** stopping at the first p < 0.05 gave a false-positive rate of 19.2% versus 5.2% for one planned look; a Bonferroni-corrected look rule gave 2.4%.

**Caveats:** unresolved sample-ratio flag; only retention and rounds measured, so any revenue or monetisation gain from the later gate is unknown; no novelty or day-of-week checks; baseline and MDE were assumptions.
