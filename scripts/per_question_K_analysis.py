"""Per-question K → per-question agent benefit analysis.

도메인-레벨 K로 8 셀을 검증하는 것보다 훨씬 강력한 검증:
각 문제마다 K(LLM judge 점수) + 각 문제의 1-agent/2-agent/4-agent 정답 여부.
질문 단위 회귀로 K가 multi-agent benefit을 예측하는가?

데이터 매칭:
  - LLM judge가 채점한 문제: feature_judge/{judge}_question-level_{domain}_*.json
    (각 항목에 item_id 포함, 250개 문제 × 2 judges)
  - 각 문제의 정답 여부: results/merged/*.json (도메인×model×agent_count별 question_id별)

분석:
  1) 각 문제의 K = (gpt_score + gemini_score) / 2
  2) 각 문제의 agent_benefit_2_1 = is_correct(N=2) - is_correct(N=1)  ∈ {-1, 0, +1}
     (model 평균 또는 model별)
  3) K vs agent_benefit 관계:
     - 점/scatter
     - K 분위 5등분(quintile)별 평균 benefit + 95% CI
     - 로지스틱 회귀: P(N=2가 N=1보다 좋음) ~ K
     - Spearman rho

산출:
  analysis/per_question_kbenefit.csv
  analysis/per_question_kbenefit_summary.json
  paper/latex_v2/figures/fig5_per_question_kbenefit.pdf
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
from collections import defaultdict
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
J = A / 'feature_judge'
M = ROOT / 'results' / 'merged'
FIG = ROOT / 'paper' / 'latex_v2' / 'figures'

DOMAINS = ['medical', 'legal', 'logic', 'long_context']  # exclude code (contaminated)
MODELS = ['gpt_4o', 'claude_sonnet_4_6']
DOMAIN_COLORS = {
    'medical': '#d62728',
    'legal': '#1f77b4',
    'logic': '#9467bd',
    'long_context': '#ff7f0e',
}

# ===== Load Q-level K scores =====
def load_qlevel_K(judge: str, domain: str) -> dict[str, float]:
    """Returns dict question_id -> K score."""
    files = sorted(J.glob(f'{judge}_question-level_{domain}_*.json'))
    if not files: return {}
    d = json.load(open(files[-1], encoding='utf-8'))
    return {r.get('item_id'): r['parsed']['K']
             for r in d if r.get('parsed') and r.get('item_id')}

# ===== Load per-question correctness from a result file =====
def load_correctness(domain: str, model: str, agents: int) -> dict[str, bool]:
    """Returns dict question_id -> is_correct for that cell."""
    for suffix in (f'{agents}agent.json', f'{agents}agent_parallel.json'):
        p = M / f'{domain}_{model}_{suffix}'
        if p.exists():
            d = json.load(open(p, encoding='utf-8'))
            out = {}
            for r in d.get('results', []):
                qid = r.get('question_id')
                if qid is None: continue
                out[qid] = bool(r.get('is_correct'))
            return out
    return {}

# ===== Build the per-question table =====
print('Loading Q-level K scores from both judges...')
K_by_judge_domain = {}
for j in ['gpt', 'gemini']:
    K_by_judge_domain[j] = {}
    for dom in DOMAINS:
        K_by_judge_domain[j][dom] = load_qlevel_K(j, dom)
        print(f'  {j}/{dom}: {len(K_by_judge_domain[j][dom])} K scores')

# Build dataframe of (domain, item_id, K_avg, is_correct_N1, is_correct_N2, ...)
rows = []
for dom in DOMAINS:
    # The 50 sample item_ids are the intersection of both judges
    K_gpt = K_by_judge_domain['gpt'].get(dom, {})
    K_gem = K_by_judge_domain['gemini'].get(dom, {})
    common = set(K_gpt) & set(K_gem)
    print(f'\n{dom}: {len(common)} common-judged items')

    # Load correctness for all model×agent combinations
    correctness = {}  # (model, agents) -> {qid: bool}
    for model in MODELS:
        for a in (1, 2, 3, 4):
            correctness[(model, a)] = load_correctness(dom, model, a)

    for qid in common:
        K = (K_gpt[qid] + K_gem[qid]) / 2
        K_disagree = abs(K_gpt[qid] - K_gem[qid])
        for model in MODELS:
            ic = {a: correctness[(model, a)].get(qid) for a in (1, 2, 3, 4)}
            # only include if at least N=1 and N=2 are present (need them for benefit)
            if ic[1] is None or ic[2] is None: continue
            rows.append({
                'domain': dom, 'model': model, 'item_id': qid,
                'K': K, 'K_gpt': K_gpt[qid], 'K_gem': K_gem[qid], 'K_disagree': K_disagree,
                'ic_1': ic[1], 'ic_2': ic[2], 'ic_3': ic[3], 'ic_4': ic[4],
                'benefit_2_1': int(ic[2]) - int(ic[1]) if (ic[1] is not None and ic[2] is not None) else None,
                'benefit_4_1': int(ic[4]) - int(ic[1]) if (ic[1] is not None and ic[4] is not None) else None,
            })

df = pd.DataFrame(rows)
print(f'\nDataframe rows: {len(df)} (domain × model × question)')
print(df.groupby(['domain', 'model']).size().to_string())

# Save raw
df.to_csv(A / 'per_question_kbenefit.csv', index=False)

# ===== Analysis 1: K vs benefit_2_1 (paired transition) =====
sub21 = df.dropna(subset=['benefit_2_1'])
print(f'\n=== Analysis: K vs benefit_2_1 (n={len(sub21)}) ===')

# overall correlation
rho, p = stats.spearmanr(sub21['K'], sub21['benefit_2_1'])
print(f'Overall Spearman rho(K, benefit_2_1) = {rho:.3f}, p = {p:.2e}')

# Quintile binning
sub21 = sub21.copy()
sub21['K_quintile'] = pd.qcut(sub21['K'], 5, labels=False, duplicates='drop')
quint_summary = sub21.groupby('K_quintile').agg(
    n=('benefit_2_1', 'size'),
    K_mean=('K', 'mean'),
    benefit_mean=('benefit_2_1', 'mean'),
    benefit_std=('benefit_2_1', 'std'),
)
quint_summary['benefit_se'] = quint_summary['benefit_std'] / np.sqrt(quint_summary['n'])
print('\nK quintiles vs benefit_2_1:')
print(quint_summary.round(3).to_string())

# Per-domain
print('\nPer-domain Spearman rho(K, benefit_2_1):')
for (dom, model), sub in sub21.groupby(['domain', 'model']):
    if len(sub) < 5: continue
    rho_d, p_d = stats.spearmanr(sub['K'], sub['benefit_2_1'])
    print(f'  {dom:14s} × {model:20s}: rho = {rho_d:+.3f}, p = {p_d:.3f}, n = {len(sub)}')

# ===== Analysis 2: Logistic — P(2-agent strictly better) ~ K =====
sub21 = sub21.copy()
# label: 1 if going to N=2 strictly improves correctness, 0 if same or hurts
sub21['improved'] = (sub21['benefit_2_1'] > 0).astype(int)
sub21['unchanged_correct'] = ((sub21['benefit_2_1'] == 0) & (sub21['ic_1'])).astype(int)
sub21['hurt'] = (sub21['benefit_2_1'] < 0).astype(int)

# Manual logistic regression
from scipy.optimize import minimize
def fit_logistic(X, y):
    X = np.asarray(X); y = np.asarray(y)
    if X.ndim == 1: X = X.reshape(-1, 1)
    Xc = np.column_stack([np.ones(len(X)), X])
    def neg_ll(beta):
        z = np.clip(Xc @ beta, -30, 30)
        p = 1/(1+np.exp(-z))
        eps = 1e-9
        return -np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    res = minimize(neg_ll, np.zeros(Xc.shape[1]), method='BFGS')
    return res.x

# logistic for hurt vs not-hurt
print('\n=== Logistic regression: P(harm) = P(2-agent worse than 1) ~ K ===')
beta = fit_logistic(sub21['K'], sub21['hurt'])
print(f'  intercept: {beta[0]:+.3f}, slope (K): {beta[1]:+.3f}')
print(f'  Higher K → higher P(harm)? {"YES" if beta[1] > 0 else "NO"}')

# logistic for improvement
beta_imp = fit_logistic(sub21['K'], sub21['improved'])
print('\n=== Logistic regression: P(2-agent strictly better than 1) ~ K ===')
print(f'  intercept: {beta_imp[0]:+.3f}, slope (K): {beta_imp[1]:+.3f}')
print(f'  Higher K → lower P(improvement)? {"YES" if beta_imp[1] < 0 else "NO"}')

# ===== Save summary =====
summary = {
    'n_total': int(len(sub21)),
    'spearman_K_benefit_2_1': {'rho': float(rho), 'p': float(p)},
    'quintile_summary': quint_summary.reset_index().to_dict('records'),
    'logistic_harm': {'intercept': float(beta[0]), 'slope_K': float(beta[1])},
    'logistic_improvement': {'intercept': float(beta_imp[0]), 'slope_K': float(beta_imp[1])},
    'per_domain_model_rho': {
        f'{dom}/{model}': float(stats.spearmanr(sub['K'], sub['benefit_2_1']).statistic)
        for (dom, model), sub in sub21.groupby(['domain', 'model']) if len(sub) >= 5
    },
}
(A / 'per_question_kbenefit_summary.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

# ===== Figure =====
matplotlib.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 110, 'savefig.dpi': 200,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax_quint, ax_scatter, ax_logit = axes

# (1) Quintile bar chart
ax = ax_quint
xs = quint_summary['K_mean'].to_numpy()
ys = quint_summary['benefit_mean'].to_numpy()
yerr = 1.96 * quint_summary['benefit_se'].to_numpy()
ax.errorbar(xs, ys, yerr=yerr, fmt='o-', color='#1f77b4', capsize=4, markersize=8, lw=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax.set_xlabel('K quintile mean')
ax.set_ylabel('Mean benefit  (acc(N=2) − acc(N=1))')
ax.set_title(f'(a) K quintile vs benefit (n={len(sub21)} items)')
ax.grid(alpha=0.3)
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.annotate(f'{int(quint_summary["n"].iloc[i])}', (x, y),
                 textcoords='offset points', xytext=(8, 8), fontsize=8, color='#555')

# (2) Scatter colored by domain
ax = ax_scatter
for dom in DOMAINS:
    sub_d = sub21[sub21['domain'] == dom]
    if len(sub_d) == 0: continue
    # jitter benefit by ±0.05 vertically for visibility
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.06, 0.06, size=len(sub_d))
    ax.scatter(sub_d['K'], sub_d['benefit_2_1'] + jitter,
                color=DOMAIN_COLORS[dom], alpha=0.6, s=18, label=dom, edgecolor='white', linewidth=0.3)
ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax.set_xlabel('K (per-question, judge mean)')
ax.set_ylabel('benefit_2_1 (jittered)')
ax.set_title(f'(b) Per-question K vs benefit\nSpearman ρ={rho:.2f} (p={p:.1e})')
ax.legend(loc='lower left', fontsize=8)
ax.grid(alpha=0.3)

# (3) Logistic fit visualisation
ax = ax_logit
K_grid = np.linspace(0, 1, 101)
def logistic(b0, b1, x):
    z = np.clip(b0 + b1 * x, -30, 30)
    return 1/(1+np.exp(-z))
ax.plot(K_grid, logistic(beta[0], beta[1], K_grid),
         color='#d62728', label=f'P(harm) — slope {beta[1]:+.2f}', lw=2)
ax.plot(K_grid, logistic(beta_imp[0], beta_imp[1], K_grid),
         color='#2ca02c', label=f'P(improve) — slope {beta_imp[1]:+.2f}', lw=2)
ax.axvline(0.6, color='purple', linestyle=':', alpha=0.6, label='τ = 0.6')
ax.set_xlabel('K')
ax.set_ylabel('Probability')
ax.set_title('(c) Logistic fits')
ax.legend(loc='best', fontsize=9)
ax.set_ylim(0, 0.6)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.suptitle(f'Per-question K predicts per-question agent benefit (n={len(sub21)})', y=1.02, fontsize=12, fontweight='bold')
out = FIG / 'fig5_per_question_kbenefit.pdf'
plt.savefig(out, bbox_inches='tight')
plt.savefig(FIG / 'fig5_per_question_kbenefit.png', bbox_inches='tight')
plt.close()
print(f'\nWrote {out}')
print(f'Wrote {A / "per_question_kbenefit.csv"} ({len(df)} rows)')
print(f'Wrote {A / "per_question_kbenefit_summary.json"}')
