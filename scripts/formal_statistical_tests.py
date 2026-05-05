"""공식 통계 검증 — paired McNemar + bootstrap CIs per (domain, model, agent_count) cell.

각 (도메인, 모델) curve에 대해:
  1) 각 N에서 accuracy + bootstrap 95% CI
  2) Paired McNemar tests on shared question_ids:
     - N=1 vs N=2
     - N=1 vs N=4
     - N=2 vs N=4
  3) Friedman test (multiple paired comparisons across N=1,2,3,4)
  4) Cohen's h effect sizes (proportion difference)

산출:
  analysis/formal_stats_per_cell.csv       — per-cell summary
  analysis/formal_stats_pairwise.csv       — pairwise N comparisons
  paper/latex_v2/figures/fig7_paired_tests.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'analysis'
M = ROOT / 'results' / 'merged'
FIG = ROOT / 'paper' / 'latex_v2' / 'figures'

matplotlib.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200})

DOMAIN_SIZES = {'medical': 5000, 'legal': 3000, 'logic': 1500, 'long_context': 902}
DOMAINS = list(DOMAIN_SIZES)
MODELS = ['gpt_4o', 'claude_sonnet_4_6']

# Tainted cells excluded from headline analysis
EXCLUDED = {('medical', 'claude_sonnet_4_6', 1),
             ('logic', 'gpt_4o', 4)}

DOMAIN_COLORS = {
    'medical': '#d62728',
    'legal': '#1f77b4',
    'logic': '#9467bd',
    'long_context': '#ff7f0e',
}


def load_cell(domain, model, agents):
    """Returns dict question_id -> bool is_correct, or None if missing/excluded."""
    if (domain, model, agents) in EXCLUDED: return None
    for suffix in (f'{agents}agent.json', f'{agents}agent_parallel.json'):
        p = M / f'{domain}_{model}_{suffix}'
        if p.exists():
            d = json.load(open(p, encoding='utf-8'))
            results = d.get('results', [])
            # dedup last
            out = {}
            for r in results:
                qid = r.get('question_id')
                if qid is None: continue
                out[qid] = bool(r.get('is_correct'))
            # exclude if too small
            if len(out) < 800: return None
            return out
    return None


def bootstrap_ci(arr, n_boot=2000, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0: return (None, None, None)
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar binomial. b/c = discordant counts."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return float(stats.binomtest(k, n, p=0.5, alternative='two-sided').pvalue)


def cohen_h(p1, p2):
    p1 = max(min(p1, 1.0), 0.0); p2 = max(min(p2, 1.0), 0.0)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


# ===== Per-cell summary =====
rng = np.random.default_rng(42)
cell_rows = []
for dom in DOMAINS:
    for model in MODELS:
        for a in (1, 2, 3, 4):
            d = load_cell(dom, model, a)
            if d is None:
                cell_rows.append({'domain': dom, 'model': model, 'agents': a,
                                   'n': 0, 'acc': None, 'ci_lo': None, 'ci_hi': None})
                continue
            arr = np.array([int(v) for v in d.values()])
            mean, lo, hi = bootstrap_ci(arr, n_boot=2000, rng=rng)
            cell_rows.append({'domain': dom, 'model': model, 'agents': a,
                               'n': len(arr), 'acc': mean, 'ci_lo': lo, 'ci_hi': hi})
cell_df = pd.DataFrame(cell_rows)
cell_df.to_csv(A / 'formal_stats_per_cell.csv', index=False)

print('=== Per-cell accuracy + 95% bootstrap CI ===')
print(cell_df.dropna(subset=['acc']).round(4).to_string(index=False))

# ===== Pairwise comparisons within each (domain, model) =====
pair_rows = []
for dom in DOMAINS:
    for model in MODELS:
        cells = {a: load_cell(dom, model, a) for a in (1, 2, 3, 4)}
        for a1, a2 in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
            d1, d2 = cells.get(a1), cells.get(a2)
            if d1 is None or d2 is None: continue
            common = set(d1) & set(d2)
            if len(common) < 100: continue
            # contingency: b = correct in d1 only, c = correct in d2 only
            b = sum(1 for q in common if d1[q] and not d2[q])
            c = sum(1 for q in common if not d1[q] and d2[q])
            both = sum(1 for q in common if d1[q] and d2[q])
            none = sum(1 for q in common if not d1[q] and not d2[q])
            p_mc = mcnemar_exact(b, c)
            acc1 = (b + both) / len(common)
            acc2 = (c + both) / len(common)
            h = cohen_h(acc2, acc1)
            # 95% CI for delta via bootstrap on paired indicators
            arr1 = np.array([int(d1[q]) for q in common])
            arr2 = np.array([int(d2[q]) for q in common])
            n = len(common)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, n, size=n)
                boots.append(arr2[idx].mean() - arr1[idx].mean())
            delta_ci_lo = float(np.percentile(boots, 2.5))
            delta_ci_hi = float(np.percentile(boots, 97.5))
            pair_rows.append({
                'domain': dom, 'model': model,
                'a1': a1, 'a2': a2, 'n_paired': len(common),
                'acc_a1': acc1, 'acc_a2': acc2,
                'delta_a2_a1': acc2 - acc1,
                'delta_ci_lo': delta_ci_lo, 'delta_ci_hi': delta_ci_hi,
                'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p_mc,
                'cohen_h': float(h),
            })
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(A / 'formal_stats_pairwise.csv', index=False)

print('\n=== Pairwise comparisons (paired on shared question_ids) ===')
print(pair_df[['domain', 'model', 'a1', 'a2', 'n_paired',
                'acc_a1', 'acc_a2', 'delta_a2_a1', 'mcnemar_p', 'cohen_h']].round(4).to_string(index=False))

# ===== Specific N=1 vs N=4 summary =====
print('\n=== Headline: N=1 vs N=4 ===')
n14 = pair_df[(pair_df['a1'] == 1) & (pair_df['a2'] == 4)]
print(n14[['domain', 'model', 'n_paired', 'acc_a1', 'acc_a2',
            'delta_a2_a1', 'delta_ci_lo', 'delta_ci_hi', 'mcnemar_p', 'cohen_h']]
       .round(4).to_string(index=False))

# ===== Friedman test per (domain, model) =====
print('\n=== Friedman test (N=1,2,3,4 paired) per (domain, model) ===')
friedman = []
for dom in DOMAINS:
    for model in MODELS:
        cells = {a: load_cell(dom, model, a) for a in (1, 2, 3, 4)}
        if any(v is None for v in cells.values()): continue
        common = set.intersection(*[set(c) for c in cells.values()])
        if len(common) < 100: continue
        cols = []
        for a in (1, 2, 3, 4):
            cols.append([int(cells[a][q]) for q in common])
        arr = np.array(cols)  # 4 × n_common
        try:
            stat, p = stats.friedmanchisquare(*arr.tolist())
            friedman.append({'domain': dom, 'model': model, 'n_paired': len(common),
                              'friedman_stat': float(stat), 'friedman_p': float(p)})
            print(f'  {dom:14s} × {model:20s}: chi2 = {stat:7.3f}, p = {p:.2e}, n = {len(common)}')
        except Exception as e:
            print(f'  {dom}/{model}: Friedman failed ({e})')

pd.DataFrame(friedman).to_csv(A / 'formal_stats_friedman.csv', index=False)

# ===== Figure: forest plot of N=1 vs N=4 deltas =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) per-cell accuracy with CIs (line plot per (dom,model))
ax = axes[0]
for (dom, model), sub in cell_df.groupby(['domain', 'model']):
    sub = sub.dropna(subset=['acc']).sort_values('agents')
    if len(sub) < 2: continue
    color = DOMAIN_COLORS[dom]
    style = '-' if model == 'gpt_4o' else '--'
    label = f'{dom} ({"GPT" if model == "gpt_4o" else "Claude"})'
    ax.plot(sub['agents'], sub['acc'] * 100, style + 'o', color=color, lw=1.8, markersize=5,
             label=label)
    ax.fill_between(sub['agents'],
                     (sub['ci_lo'] * 100).astype(float),
                     (sub['ci_hi'] * 100).astype(float),
                     color=color, alpha=0.10)
ax.set_xlabel('Number of agents (N)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('(a) Per-cell accuracy with 95% bootstrap CI')
ax.set_xticks([1, 2, 3, 4])
ax.legend(loc='best', fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# (b) Forest plot for N=1 vs N=4
ax = axes[1]
n14_sorted = n14.sort_values('delta_a2_a1').reset_index(drop=True)
y_positions = np.arange(len(n14_sorted))
for i, r in n14_sorted.iterrows():
    color = DOMAIN_COLORS[r['domain']]
    ax.errorbar(r['delta_a2_a1'] * 100, i,
                 xerr=[[(r['delta_a2_a1'] - r['delta_ci_lo']) * 100],
                       [(r['delta_ci_hi'] - r['delta_a2_a1']) * 100]],
                 fmt='s', color=color, capsize=4, markersize=10, ecolor=color, lw=2)
    label = f"{r['domain']} ({'GPT' if r['model'] == 'gpt_4o' else 'Claude'})"
    sig = ''
    if r['mcnemar_p'] < 0.001: sig = '***'
    elif r['mcnemar_p'] < 0.01: sig = '**'
    elif r['mcnemar_p'] < 0.05: sig = '*'
    ax.text(r['delta_ci_hi'] * 100 + 1, i, f'{label} {sig}', va='center', fontsize=9, color=color)
ax.axvline(0, color='gray', linestyle='--')
ax.set_yticks([])
ax.set_xlabel('Δ accuracy: acc(N=4) − acc(N=1) (pp)')
ax.set_title('(b) N=1 vs N=4 paired comparison\n(McNemar significance: ***p<0.001, **p<0.01, *p<0.05)')
ax.grid(alpha=0.3, axis='x')
ax.set_xlim(-40, 25)

plt.tight_layout()
fig.suptitle('Formal statistical validation', y=1.02, fontsize=12, fontweight='bold')
out = FIG / 'fig7_paired_tests.pdf'
plt.savefig(out, bbox_inches='tight')
plt.savefig(FIG / 'fig7_paired_tests.png', bbox_inches='tight')
plt.close()
print(f'\nWrote {out}')
print(f'Wrote {A / "formal_stats_per_cell.csv"}')
print(f'Wrote {A / "formal_stats_pairwise.csv"}')
print(f'Wrote {A / "formal_stats_friedman.csv"}')
