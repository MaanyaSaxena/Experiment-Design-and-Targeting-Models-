# Method justification

| Method | Why used | Key assumptions | If assumption fails |
|---|---|---|---|
| Power / sample-size calc | Fix n and MDE before looking at outcomes | Normal approximation; baseline rate known | Simulate power; use exact tests for small n |
| Sample-ratio check (chi-square) | Detect broken randomisation or logging | Planned split known (50/50) | Investigate assignment/logging; do not analyse until explained |
| Two-proportion z-test | Compare binary retention | Independent users; large n | Exact/Fisher test, or cluster-robust methods |
| Bootstrap CI | Interval without a normal formula | Users are i.i.d. | Cluster bootstrap |
| Permutation test (game rounds) | Rounds are heavy-tailed | Exchangeability under H0 (given by randomisation) | Mann-Whitney / trimmed or log-scale analysis |
| Holm correction | Control family-wise error across metrics | Metrics pre-declared | BH if many exploratory metrics |
| Peeking simulation | Show repeated looks inflate Type I error | A/A simulation mirrors real stopping rule | Use group-sequential or alpha-spending designs |
| Stratified k-fold CV | Preserve ~11% positive rate per fold | Rows independent | Group k-fold if customers repeat |
| PR-AUC, recall/lift at top 10% | Imbalanced data, limited call budget | Top-10% reflects actual capacity | Set the cutoff from the real call budget |
| Logistic regression | Interpretable baseline | Roughly linear log-odds | Add interactions/splines or use trees |
| KNN | Distance-based comparison | Scaled features; sensible k; distance meaningful | PCA / feature selection; limited by high-dimensional one-hot space and slow prediction |
| Gradient boosting | Captures nonlinearity and interactions | Enough data; tuned regularisation | Stronger regularisation, simpler model |
| Paired bootstrap on differences | Test whether models differ, not eyeball AUC | Same rows scored by both models | Repeated CV with corrected t-test; nested CV |
