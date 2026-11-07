# Decision memo: Calling list
All numbers are taken from the final executed notebook (bank-additional-full.csv, 41,188 rows, 11.3% subscribe). `duration` is excluded because it is only known after the call.

**Question:** Which customers should the team call first, given limited agent time?

**Random 5-fold stratified CV (same folds for all models)**
| Model | PR-AUC | Recall @ top 10% | Lift @ top 10% |
|---|---|---|---|
| Logistic regression | 0.451 | 0.436 | 4.36 |
| KNN (k = 200) | 0.383 | 0.413 | 4.13 |
| Gradient boosting | 0.469 | 0.457 | 4.57 |

**Paired bootstrap, difference (95% CI), test-row noise only**
| Comparison | PR-AUC | Recall @ 10% |
|---|---|---|
| GB − LR | +0.018 (0.011 to 0.025) | +0.021 (0.013 to 0.029) |
| GB − KNN | +0.086 (0.076 to 0.095) | +0.044 (0.035 to 0.053) |
Under random CV, gradient boosting is ahead of logistic regression by a small but consistent margin, and both clearly beat KNN.

**Leakage check (gradient boosting):** with `duration` PR-AUC 0.668, recall@10% 0.582; without it 0.469 and 0.457. Keeping `duration` would have inflated the scores by about 0.20 PR-AUC.

**Macro indicators dropped (random CV, PR-AUC):** GB 0.469 → 0.422, LR 0.451 → 0.402, KNN 0.383 → 0.334. Model ranking unchanged; about 0.05 PR-AUC comes from the macro columns, which mostly identify when a call happened.

**Time-ordered split (train earliest 70%, test latest 30%)**
Test-period subscribe rate is 24.55% versus 11.3% overall, so PR-AUC is not comparable with the CV table; use lift and ROC-AUC.
| Model | PR-AUC | ROC-AUC | Recall @ 10% | Lift @ 10% |
|---|---|---|---|---|
| Logistic regression | 0.391 | 0.667 | 0.188 | 1.88 |
| KNN | 0.227 | 0.491 | 0.065 | 0.65 |
| Gradient boosting | 0.312 | 0.569 | 0.156 | 1.56 |
Performance falls sharply and the ranking changes: logistic regression is best, and KNN is no better than chance.

**Calling-list guidance (random CV, gradient boosting):** calling the top 10% of scores reaches 45.7% of subscribers (the top decile converts at about 52%); the top 20% reaches 64.8%; the top 30% reaches 72.2%.

**Recommendation: use logistic regression for deployment.** Gradient boosting wins under random CV, but the time-ordered split is closer to real use and favours logistic regression, which is also easier to explain and monitor. Rank customers by predicted probability and call from the top; where to stop depends on call cost, which the data lacks.
**Caveats:** the time-split result is one split with no CI and its cause is untested (rolling windows and a no-macro time split were not run); one Portuguese bank, 2008-2010; no call-cost data, so the calling cut-off cannot be set economically; paired CIs ignore retraining variability; k = 200 sits at the edge of the tested grid.
