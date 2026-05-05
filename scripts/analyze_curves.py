"""곡선 형태 분류 + 통계 검증.

각 (domain, model) curve에 대해:
- Spearman ρ(agent_count, is_correct) → 단조성 + 방향
- Mann-Kendall trend test
- Peak agent count
- Bootstrap 95% CI for accuracy at each agent count
- McNemar 1-agent vs 4-agent (paired on same question_ids)
- Cohen's h effect size (1 vs 4-agent)
- Curve shape 분류: decreasing / increasing / inverted-U / flat / unclear

입력:
  analysis/per_question.csv
  analysis/accuracy_table.csv
출력:
  analysis/curve_classification.csv
  analysis/pairwise_tests.csv
  analysis/bootstrap_ci.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / 'analysis'

RNG = np.random.default_rng(42)
N_BOOT = 2000
ALPHA = 0.05

# 셀이 너무 작아서 신뢰할 수 없는 한계 (재실험 필요한 셀 식별 후 분석에서 *제외*하지 않고 *플래그*만)
MIN_RELIABLE_N = 800

def cohen_h(p1: float, p2: float) -> float:
    """Effect size for difference of two proportions."""
    p1 = max(min(p1, 1.0), 0.0)
    p2 = max(min(p2, 1.0), 0.0)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Returns (S, p_value) two-sided. x is the sequence of values along ordered index."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return s, p

def bootstrap_ci_proportion(correct: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for proportion."""
    if len(correct) == 0:
        return (np.nan, np.nan)
    boots = RNG.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 100 * ALPHA / 2)), float(np.percentile(boots, 100 * (1 - ALPHA / 2)))

def mcnemar(b: int, c: int) -> float:
    """Exact McNemar test (two-sided). b = correct in cond1 only, c = correct in cond2 only."""
    n = b + c
    if n == 0:
        return 1.0
    # binomial test (two-sided) under H0: p=0.5
    k = min(b, c)
    return float(stats.binomtest(k, n, p=0.5, alternative='two-sided').pvalue)

def classify_shape(spearman_rho: float, sp_p: float, peak: int,
                    accs: dict[int, float]) -> str:
    """Curve shape classification.
    Inputs:
      spearman_rho, sp_p — Spearman over (agent_count, accuracy)
      peak — argmax agent_count
      accs — dict of agent_count → accuracy (proportion 0-1)
    """
    diffs = [accs[i+1] - accs[i] for i in (1, 2, 3) if (i in accs and i+1 in accs)]
    if not diffs:
        return 'insufficient'

    n_pos = sum(1 for d in diffs if d > 0.005)
    n_neg = sum(1 for d in diffs if d < -0.005)
    n_flat = sum(1 for d in diffs if -0.005 <= d <= 0.005)

    # Strict monotone: spearman significant + |rho| > 0.8
    if sp_p < ALPHA and spearman_rho < -0.8:
        return 'decreasing'
    if sp_p < ALPHA and spearman_rho > 0.8:
        return 'increasing'

    # Peak in middle suggests inverted-U
    if peak in (2, 3) and n_pos >= 1 and n_neg >= 1:
        return 'inverted_U'

    # Mostly flat
    if n_flat >= 2 and abs(diffs[0] + diffs[-1]) < 0.02:
        return 'flat'

    # Sign-based
    if n_neg >= 2 and n_pos == 0:
        return 'decreasing_weak'
    if n_pos >= 2 and n_neg == 0:
        return 'increasing_weak'

    return 'unclear'

def main():
    pq = pd.read_csv(ANALYSIS / 'per_question.csv')
    summary = pd.read_csv(ANALYSIS / 'accuracy_table.csv')

    # 분석 대상: (domain, model) pair 중 agent count 1-4가 어느 정도 다 있는 것만
    classification_rows = []
    pairwise_rows = []
    bootstrap_rows = []

    for (dom, model), sub in pq.groupby(['domain', 'model']):
        # build per-agent set
        agents_present = sorted(sub['agent_count'].unique().tolist())
        # require at least 1, 2, 3, 4 (some Gemini will fail this)
        if len(agents_present) < 4:
            continue
        # require 1, 2, 3, 4 all
        if set(agents_present) != {1, 2, 3, 4}:
            continue

        accs: dict[int, float] = {}
        ns: dict[int, int] = {}
        any_unreliable = False
        unreliable_agents = []
        for a in (1, 2, 3, 4):
            cell = sub[sub['agent_count'] == a]
            arr = cell['is_correct'].astype(int).to_numpy()
            ns[a] = len(arr)
            accs[a] = float(arr.mean()) if len(arr) else np.nan
            ci_lo, ci_hi = bootstrap_ci_proportion(arr)
            if len(arr) < MIN_RELIABLE_N:
                any_unreliable = True
                unreliable_agents.append(a)
            bootstrap_rows.append({
                'domain': dom, 'model': model, 'agent_count': a,
                'n': ns[a], 'accuracy': accs[a],
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'reliable': bool(len(arr) >= MIN_RELIABLE_N),
            })

        # Spearman over the four points (rho, p approximate for n=4 so use exact alternative)
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([accs[a] for a in (1, 2, 3, 4)])
        sp = stats.spearmanr(x, y)
        sp_rho, sp_p = float(sp.statistic), float(sp.pvalue)

        # Mann-Kendall
        mk_s, mk_p = mann_kendall(y)

        peak_agent = int(max(accs, key=lambda a: accs[a]))
        worst_agent = int(min(accs, key=lambda a: accs[a]))

        shape = classify_shape(sp_rho, sp_p, peak_agent, accs)

        # 1 vs 4 paired comparison (same question_ids)
        c1 = sub[sub['agent_count'] == 1].set_index('question_id')['is_correct']
        c4 = sub[sub['agent_count'] == 4].set_index('question_id')['is_correct']
        common = c1.index.intersection(c4.index)
        c1c, c4c = c1.loc[common].astype(bool), c4.loc[common].astype(bool)
        # b: correct in 1-agent only; c: correct in 4-agent only
        b = int(((c1c) & (~c4c)).sum())
        cc = int(((~c1c) & (c4c)).sum())
        mn_p = mcnemar(b, cc)
        h = cohen_h(accs[1], accs[4])
        delta = accs[4] - accs[1]
        # bootstrap CI for delta
        boot_deltas = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, len(common), size=len(common))
            ci1 = c1c.values[idx]
            ci4 = c4c.values[idx]
            boot_deltas.append(ci4.mean() - ci1.mean())
        delta_lo, delta_hi = (float(np.percentile(boot_deltas, 100 * ALPHA / 2)),
                              float(np.percentile(boot_deltas, 100 * (1 - ALPHA / 2))))
        pairwise_rows.append({
            'domain': dom, 'model': model,
            'acc_1agent': accs[1], 'acc_4agent': accs[4],
            'delta_4_minus_1': delta,
            'delta_ci_lo': delta_lo, 'delta_ci_hi': delta_hi,
            'cohen_h': float(h),
            'mcnemar_p': float(mn_p),
            'mcnemar_b_only1': b, 'mcnemar_c_only4': cc,
            'n_paired': int(len(common)),
        })

        classification_rows.append({
            'domain': dom, 'model': model,
            'acc_1': accs[1], 'acc_2': accs[2], 'acc_3': accs[3], 'acc_4': accs[4],
            'n_min': min(ns.values()),
            'spearman_rho': sp_rho, 'spearman_p': sp_p,
            'mann_kendall_S': mk_s, 'mann_kendall_p': mk_p,
            'peak_agent': peak_agent, 'worst_agent': worst_agent,
            'shape': shape,
            'reliability_flag': '!unreliable' if any_unreliable else 'ok',
            'unreliable_agents': ','.join(map(str, unreliable_agents)) if unreliable_agents else '',
        })

    cls_df = pd.DataFrame(classification_rows).sort_values(['domain', 'model'])
    pair_df = pd.DataFrame(pairwise_rows).sort_values(['domain', 'model'])
    boot_df = pd.DataFrame(bootstrap_rows).sort_values(['domain', 'model', 'agent_count'])

    cls_df.to_csv(ANALYSIS / 'curve_classification.csv', index=False, encoding='utf-8')
    pair_df.to_csv(ANALYSIS / 'pairwise_tests.csv', index=False, encoding='utf-8')
    boot_df.to_csv(ANALYSIS / 'bootstrap_ci.csv', index=False, encoding='utf-8')

    print('=== Curve classification ===')
    cols = ['domain', 'model', 'acc_1', 'acc_2', 'acc_3', 'acc_4',
            'spearman_rho', 'spearman_p', 'peak_agent', 'shape', 'reliability_flag']
    print(cls_df[cols].round(4).to_string(index=False))
    print()

    print('=== Pairwise 1 vs 4 agent (paired) ===')
    cols2 = ['domain', 'model', 'acc_1agent', 'acc_4agent', 'delta_4_minus_1',
            'delta_ci_lo', 'delta_ci_hi', 'cohen_h', 'mcnemar_p']
    print(pair_df[cols2].round(4).to_string(index=False))
    print()

    print('=== Shape distribution ===')
    print(cls_df['shape'].value_counts().to_string())
    print()

    print(f"Wrote curve_classification.csv ({len(cls_df)} rows)")
    print(f"Wrote pairwise_tests.csv      ({len(pair_df)} rows)")
    print(f"Wrote bootstrap_ci.csv        ({len(boot_df)} rows)")

if __name__ == '__main__':
    main()
