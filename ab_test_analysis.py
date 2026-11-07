"""
ab_test_analysis.py
Reusable A/B test toolkit (numpy / scipy / pandas only).

Covers: sample-size and power, sample-ratio check, two-proportion z-test,
bootstrap CI, permutation test, multiple-testing correction, and a
peeking (Type I error inflation) simulation.

Usage as a script (Cookie Cats schema: userid, version, sum_gamerounds,
retention_1, retention_7):
    python ab_test_analysis.py --csv cookie_cats.csv --control gate_30 --treatment gate_40
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 42


# --------------------------------------------------------------------------
# 1. Power and sample size
# --------------------------------------------------------------------------
def sample_size_two_proportions(p_base, mde_abs, alpha=0.05, power=0.80,
                                two_sided=True, ratio=1.0):
    """Per-group n (control) to detect an absolute lift `mde_abs` over `p_base`.
    Normal approximation with unpooled variance under the alternative.
    `ratio` = n_treatment / n_control. Returns (n_control, n_treatment)."""
    p1, p2 = p_base, p_base + mde_abs
    z_a = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = stats.norm.ppf(power)
    var = p1 * (1 - p1) + p2 * (1 - p2) / ratio
    n1 = (z_a + z_b) ** 2 * var / mde_abs ** 2
    n1 = int(np.ceil(n1))
    return n1, int(np.ceil(n1 * ratio))


def power_two_proportions(p_base, mde_abs, n_per_group, alpha=0.05):
    """Approximate power of a two-sided two-proportion z-test."""
    p1, p2 = p_base, p_base + mde_abs
    se = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z = abs(mde_abs) / se
    return float(stats.norm.cdf(z - z_a) + stats.norm.cdf(-z - z_a))


def minimum_detectable_effect(p_base, n_per_group, alpha=0.05, power=0.80):
    """Smallest absolute lift detectable with the given n (grid search)."""
    grid = np.linspace(1e-5, 0.2, 4000)
    pw = np.array([power_two_proportions(p_base, d, n_per_group, alpha) for d in grid])
    ok = np.where(pw >= power)[0]
    return float(grid[ok[0]]) if len(ok) else np.nan


# --------------------------------------------------------------------------
# 2. Data checks
# --------------------------------------------------------------------------
def sample_ratio_check(n_control, n_treatment, expected_ratio=0.5):
    """Chi-square goodness-of-fit for the allocation. Small p => SRM problem.
    Common alert threshold is p < 0.001."""
    total = n_control + n_treatment
    exp = np.array([total * expected_ratio, total * (1 - expected_ratio)])
    chi2, p = stats.chisquare([n_control, n_treatment], exp)
    return {"chi2": float(chi2), "p_value": float(p),
            "observed_share_control": n_control / total}


def basic_data_checks(df, id_col="userid", group_col="version", value_cols=()):
    out = {
        "rows": len(df),
        "duplicate_ids": int(df[id_col].duplicated().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "group_counts": df[group_col].value_counts().to_dict(),
    }
    for c in value_cols:
        out[f"{c}_describe"] = df[c].describe(percentiles=[.5, .9, .99, .999]).to_dict()
    return out


# --------------------------------------------------------------------------
# 3. Two-proportion tests
# --------------------------------------------------------------------------
def two_proportion_ztest(x_c, n_c, x_t, n_t, alpha=0.05):
    """Pooled z-test for H0: p_t = p_c, plus unpooled Wald CI for the difference."""
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unp = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    return {"p_control": p_c, "p_treatment": p_t, "diff": diff,
            "rel_lift": diff / p_c if p_c else np.nan,
            "z": float(z), "p_value": float(p_val),
            "ci_low": diff - zc * se_unp, "ci_high": diff + zc * se_unp}


def bootstrap_diff_ci(a, b, stat=np.mean, n_boot=5000, alpha=0.05, seed=RNG_SEED):
    """Percentile bootstrap CI for stat(b) - stat(a). a=control, b=treatment."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = stat(rng.choice(b, b.size)) - stat(rng.choice(a, a.size))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(stat(b) - stat(a)), "ci_low": float(lo), "ci_high": float(hi)}


# --------------------------------------------------------------------------
# 4. Permutation test (skewed metrics)
# --------------------------------------------------------------------------
def permutation_test(a, b, stat=np.mean, n_perm=10000, seed=RNG_SEED):
    """Two-sided permutation test for stat(b) - stat(a).
    Assumes exchangeability under H0 (true under randomisation)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    obs = stat(b) - stat(a)
    pooled = np.concatenate([a, b])
    na = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = stat(pooled[na:]) - stat(pooled[:na])
        if abs(d) >= abs(obs) - 1e-12:
            count += 1
    return {"observed_diff": float(obs), "p_value": (count + 1) / (n_perm + 1)}


# --------------------------------------------------------------------------
# 5. Multiple-metric correction
# --------------------------------------------------------------------------
def adjust_pvalues(pvals, method="holm"):
    """Bonferroni, Holm, or Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * p[idx])
            adj[idx] = min(running, 1.0)
    elif method == "bh":
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            prev = min(prev, p[idx] * m / (rank + 1))
            adj[idx] = prev
    else:
        raise ValueError("method must be bonferroni, holm or bh")
    return adj


# --------------------------------------------------------------------------
# 6. Peeking simulation
# --------------------------------------------------------------------------
def simulate_peeking(p=0.45, n_total=40000, n_looks=10, alpha=0.05,
                     n_sims=2000, seed=RNG_SEED):
    """Simulate A/A tests (no true effect). Compare:
       - a single look at the planned sample size
       - stopping at the first look with p < alpha (peeking)
       - Bonferroni-corrected looks (alpha / n_looks)
    Returns empirical false-positive rates."""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_total / n_looks, n_total, n_looks).astype(int)
    peek_hits = fixed_hits = bonf_hits = 0
    for _ in range(n_sims):
        a = rng.random(n_total // 2 * 2) < p  # control stream
        b = rng.random(n_total // 2 * 2) < p  # treatment stream (same p)
        pvals = []
        for n in looks:
            k = n // 2
            r = two_proportion_ztest(a[:k].sum(), k, b[:k].sum(), k)
            pvals.append(r["p_value"])
        pvals = np.array(pvals)
        peek_hits += (pvals < alpha).any()
        fixed_hits += pvals[-1] < alpha
        bonf_hits += (pvals < alpha / n_looks).any()
    return {"n_looks": n_looks, "alpha": alpha, "n_sims": n_sims,
            "fixed_single_look_fpr": fixed_hits / n_sims,
            "peeking_fpr": peek_hits / n_sims,
            "bonferroni_looks_fpr": bonf_hits / n_sims}


# --------------------------------------------------------------------------
# 7. End-to-end runner for Cookie Cats-style data
# --------------------------------------------------------------------------
def analyse_cookie_cats(df, control="gate_30", treatment="gate_40",
                        group_col="version", seed=RNG_SEED, n_boot=5000, n_perm=10000):
    results = {}
    c = df[df[group_col] == control]
    t = df[df[group_col] == treatment]
    results["srm"] = sample_ratio_check(len(c), len(t))

    pvals, names = [], []
    for col in ["retention_1", "retention_7"]:
        cx, tx = c[col].astype(int), t[col].astype(int)
        z = two_proportion_ztest(cx.sum(), len(cx), tx.sum(), len(tx))
        bt = bootstrap_diff_ci(cx.values, tx.values, n_boot=n_boot, seed=seed)
        results[col] = {"ztest": z, "bootstrap": bt}
        pvals.append(z["p_value"]); names.append(col)

    rounds = {}
    for label, fn in [("mean", np.mean), ("median", np.median)]:
        rounds[label] = permutation_test(c["sum_gamerounds"].values,
                                         t["sum_gamerounds"].values,
                                         stat=fn, n_perm=n_perm, seed=seed)
    results["gamerounds_permutation"] = rounds
    pvals.append(rounds["mean"]["p_value"]); names.append("gamerounds_mean")
    # Mann-Whitney as a rank-based cross-check
    mw = stats.mannwhitneyu(c["sum_gamerounds"], t["sum_gamerounds"], alternative="two-sided")
    results["gamerounds_mannwhitney_p"] = float(mw.pvalue)

    results["multiple_testing"] = pd.DataFrame({
        "metric": names, "raw_p": pvals,
        "holm_p": adjust_pvalues(pvals, "holm"),
        "bonferroni_p": adjust_pvalues(pvals, "bonferroni"),
    })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--control", default="gate_30")
    ap.add_argument("--treatment", default="gate_40")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    for k, v in analyse_cookie_cats(df, args.control, args.treatment).items():
        print(f"\n== {k} ==\n{v}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# 8. Outlier sensitivity for a skewed metric
# --------------------------------------------------------------------------
def outlier_sensitivity(df, col="sum_gamerounds", group_col="version",
                        control="gate_30", treatment="gate_40",
                        drop_top_k=(0, 1, 5), winsor_q=0.999,
                        n_perm=5000, seed=RNG_SEED):
    """Re-run the mean-difference permutation test under different outlier rules:
    drop the k largest values overall, winsorise at a quantile, and log1p.
    Shows whether a mean difference is driven by a handful of extreme users."""
    rows = []

    def run(label, data, transform=None):
        c = data.loc[data[group_col] == control, col].values.astype(float)
        t = data.loc[data[group_col] == treatment, col].values.astype(float)
        if transform is not None:
            c, t = transform(c), transform(t)
        r = permutation_test(c, t, stat=np.mean, n_perm=n_perm, seed=seed)
        rows.append({"rule": label, "n_control": len(c), "n_treatment": len(t),
                     "mean_diff": r["observed_diff"], "perm_p": r["p_value"]})

    for k in drop_top_k:
        data = df.drop(df[col].nlargest(k).index) if k else df
        run("as is" if k == 0 else f"drop top {k}", data)
    cap = df[col].quantile(winsor_q)
    run(f"winsorise at p{winsor_q * 100:g}", df, lambda v: np.minimum(v, cap))
    run("log1p", df, np.log1p)
    return pd.DataFrame(rows)
