"""
targeting_models.py
Part B helpers: UCI Bank Marketing (bank-full.csv or bank-additional-full.csv, sep=';').

Design choices
- `duration` is dropped: call length is only known after the call (leakage).
- Preprocessing lives inside sklearn Pipelines so scaling/encoding is fit
  on training folds only.
- All three models are scored on the SAME stratified folds, so per-fold
  and out-of-fold predictions can be compared in a paired way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
LEAKY = ["duration"]


def load_bank(path, sep=";"):
    df = pd.read_csv(path, sep=sep)
    df["y"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
    return df


def split_columns(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return num, cat


def make_preprocessor(num, cat, scale=True):
    num_t = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_t, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def build_models(num, cat, k=25, use_pca=False, pca_components=0.95):
    knn_steps = [("prep", make_preprocessor(num, cat, scale=True))]
    if use_pca:
        knn_steps.append(("pca", PCA(n_components=pca_components, random_state=SEED)))
    knn_steps.append(("clf", KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)))
    return {
        "logreg": Pipeline([("prep", make_preprocessor(num, cat, True)),
                            ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "knn": Pipeline(knn_steps),
        "gboost": Pipeline([("prep", make_preprocessor(num, cat, False)),
                            ("clf", HistGradientBoostingClassifier(
                                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                l2_regularization=1.0, random_state=SEED))]),
    }


def tune_knn_k(X, y, num, cat, k_grid=(5, 15, 25, 50, 100, 200), cv=5, use_pca=False):
    """Choose k by average precision inside a CV loop (use on training data only)."""
    pipe = build_models(num, cat, use_pca=use_pca)["knn"]
    gs = GridSearchCV(pipe, {"clf__n_neighbors": list(k_grid)},
                      scoring="average_precision",
                      cv=StratifiedKFold(cv, shuffle=True, random_state=SEED), n_jobs=1)
    gs.fit(X, y)
    return gs.best_params_["clf__n_neighbors"], pd.DataFrame(gs.cv_results_)


# -------------------------- metrics --------------------------------------
def recall_at_top_frac(y_true, scores, frac=0.10):
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * frac)))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return y_true[top].sum() / y_true.sum()


def lift_at_top_frac(y_true, scores, frac=0.10):
    return recall_at_top_frac(y_true, scores, frac) / frac


def metric_row(y, s):
    return {"pr_auc": average_precision_score(y, s),
            "roc_auc": roc_auc_score(y, s),
            "recall@10%": recall_at_top_frac(y, s, 0.10),
            "lift@10%": lift_at_top_frac(y, s, 0.10)}


# -------------------------- CV + paired comparison -------------------------
def oof_scores(models, X, y, n_splits=5, seed=SEED):
    """Out-of-fold predicted probabilities on identical stratified folds."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    out = {}
    for name, m in models.items():
        out[name] = cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]
    return out


def per_fold_metrics(models, X, y, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        for name, m in models.items():
            m.fit(X.iloc[tr], y.iloc[tr])
            s = m.predict_proba(X.iloc[te])[:, 1]
            rows.append({"fold": fold, "model": name, **metric_row(y.iloc[te], s)})
    return pd.DataFrame(rows)


def paired_bootstrap_diff(y, s_a, s_b, metric="pr_auc", n_boot=1000, alpha=0.05, seed=SEED):
    """Bootstrap CI for metric(A) - metric(B), resampling the SAME rows for both
    models. Uses out-of-fold scores, so it captures test-set sampling noise but
    not training-set variability; say so in the write-up."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    fn = {"pr_auc": average_precision_score,
          "recall@10%": lambda yy, ss: recall_at_top_frac(yy, ss, 0.10)}[metric]
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            d[i] = np.nan
            continue
        d[i] = fn(y[idx], s_a[idx]) - fn(y[idx], s_b[idx])
    d = d[~np.isnan(d)]
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(fn(y, s_a) - fn(y, s_b)), "ci_low": float(lo), "ci_high": float(hi)}


def gains_table(y, scores, n_bins=10):
    """Decile gains: what share of subscribers is captured by calling top X%."""
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores)}).sort_values("s", ascending=False)
    df["decile"] = np.floor(np.arange(len(df)) / len(df) * n_bins).astype(int) + 1
    g = df.groupby("decile")["y"].agg(["size", "sum"])
    g["cum_capture"] = g["sum"].cumsum() / g["sum"].sum()
    g["cum_call_share"] = g["size"].cumsum() / g["size"].sum()
    return g.reset_index()


def leakage_check(models_factory, X, y, leaky_cols=("duration",), n_splits=5):
    """Quantify how much leakage inflates PR-AUC: fit with and without leaky columns.
    models_factory(X) -> dict of models built for that X's columns."""
    res = {}
    for label, cols in [("without_leaky", [c for c in X.columns if c not in leaky_cols]),
                        ("with_leaky", list(X.columns))]:
        Xs = X[cols]
        ms = models_factory(Xs)
        sc = oof_scores({"gboost": ms["gboost"]}, Xs, y, n_splits)
        res[label] = metric_row(y, sc["gboost"])
    return pd.DataFrame(res).T


# -------------------------- robustness checks ------------------------------
MACRO_COLS = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


def time_split_eval(models_factory, X, y, train_frac=0.70):
    """Train on the earliest `train_frac` of rows, test on the latest rest.
    Assumes rows are in date order (true for the UCI Bank Marketing files; verify
    in the names file). Compare with random-CV scores to see if CV was optimistic.
    Returns (metrics DataFrame, test positive rate)."""
    cut = int(len(X) * train_frac)
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    rows = {}
    for name, m in models_factory(X).items():
        m.fit(Xtr, ytr)
        rows[name] = metric_row(yte, m.predict_proba(Xte)[:, 1])
    return pd.DataFrame(rows).T, float(yte.mean())


def drop_columns_comparison(models_factory, X, y, cols, n_splits=5):
    """Random stratified CV with and without `cols` (e.g. macro indicators that
    act as a campaign-time proxy). Returns a long DataFrame of metrics per model."""
    keep = [c for c in X.columns if c not in cols]
    out = []
    for label, Xs in [("all_features", X), ("without_cols", X[keep])]:
        ms = models_factory(Xs)
        sc = oof_scores(ms, Xs, y, n_splits)
        for name, s in sc.items():
            out.append({"setting": label, "model": name, **metric_row(y, s)})
    return pd.DataFrame(out)


# -------------------------- rolling time windows ---------------------------
def rolling_time_eval(models_factory, X, y, n_windows=4, initial_frac=0.5):
    """Expanding-window evaluation on date-ordered rows. Train on all rows before a
    block, test on that block, then move forward. Gives a spread of out-of-time
    results instead of one split. `pr_auc_vs_base` = PR-AUC / test positive rate,
    which is comparable across windows with different base rates."""
    n = len(X)
    start = int(n * initial_frac)
    edges = np.linspace(start, n, n_windows + 1).astype(int)
    rows = []
    for w in range(n_windows):
        tr_end, te_end = edges[w], edges[w + 1]
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        if yte.sum() == 0:
            continue
        for name, m in models_factory(X).items():
            m.fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
            r = metric_row(yte, s)
            r["pr_auc_vs_base"] = r["pr_auc"] / yte.mean()
            rows.append({"window": w + 1, "model": name, "train_rows": tr_end,
                         "test_rows": te_end - tr_end,
                         "test_pos_rate": float(yte.mean()), **r})
    return pd.DataFrame(rows)


def paired_fold_differences(folds_df, model_a, model_b,
                            metrics=("pr_auc", "recall@10%")):
    """Per-fold difference (A - B) from `per_fold_metrics` output, with mean and SD.
    Folds are shared by both models, so this is a paired view."""
    out = []
    for m in metrics:
        p = folds_df.pivot(index="fold", columns="model", values=m)
        d = p[model_a] - p[model_b]
        out.append({"metric": m, "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
                    "min_diff": d.min(), "max_diff": d.max(),
                    "folds_a_better": int((d > 0).sum()), "n_folds": len(d)})
    return pd.DataFrame(out)
