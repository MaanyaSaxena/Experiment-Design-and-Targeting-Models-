# Experiment Design and Targeting Models

Two public-data case studies. The business problems are realistic framings, not client work.

| Part | Question | Data | Method highlights |
|---|---|---|---|
| A | Did moving a game's progression gate from level 30 to 40 change retention? | Cookie Cats (Kaggle), 90,189 players | Power analysis, sample-ratio check, z-test and bootstrap CI, permutation test, Holm correction, peeking simulation |
| B | Whom should a bank's call team contact first? | UCI Bank Marketing `bank-additional-full.csv`, 41,188 calls | Logistic regression vs KNN vs gradient boosting, stratified CV, PR-AUC and top-10% recall, paired bootstrap, leakage and time-split checks |

## Headline results
- **A:** 7-day retention fell 0.82 pp (95% CI -1.33 to -0.31, Holm p = 0.0047). Recommendation: do not ship the level-40 gate.
- **A:** stopping at the first p < 0.05 over 10 looks gives a 19.2% false-positive rate versus 5.2% for one planned look.
- **B:** `duration` is dropped because it is known only after the call; keeping it would lift PR-AUC from 0.469 to 0.668.
- **B:** gradient boosting is best under random CV (PR-AUC 0.469), but a time-ordered split reverses the ranking, so logistic regression is recommended.

## Files
| File | Purpose |
|---|---|
| `FINAL_experiment_targeting_clean.ipynb` | Full analysis with saved outputs, both parts |
| `ab_test_analysis.py` | A/B test toolkit: power, SRM, z-test, bootstrap, permutation, Holm, peeking simulation |
| `targeting_models.py` | Part B pipelines, ranking metrics, paired bootstrap, leakage and time-split checks |
| `REPORT` | Full report: objectives, hypotheses, methods, results, limitations (export from the Claude Doc as Word, PDF or Markdown and add it here) |
| `memo_part_a.md`, `memo_part_b.md` | One-page decision memos |
| `method_justification.md` | Method, assumption and fallback table |

## Run
```
pip install numpy pandas scipy scikit-learn matplotlib
```
Put `cookie_cats.csv`, `bank-additional-full.csv` and the two `.py` files next to the notebook, then run all cells. The first three cells are Colab upload cells; skip them when running locally. Seed 42.

## Limits
- Baseline retention (19%) and the 1 pp minimum detectable effect are assumptions.
- A sample-ratio flag (p = 0.0086) is unresolved, and the effect of one extreme player on the game-rounds result was not tested.
- The time-split result is a single split without a confidence interval; its cause is untested.
- Rolling-window, per-fold and wider-KNN-grid checks exist in the code but were not run.
- One Portuguese bank, 2008 to 2010; no call-cost data.
