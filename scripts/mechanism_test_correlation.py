"""Mechanism test — voting asymmetry decomposition.

Result JSON에 per-agent trace가 없어 직접 error correlation 측정은 불가.
대신 *flip pattern* 분해로 ensemble dynamics를 정량화.

핵심 식 (deterministic):
  Δ_acc(N=1 → N) = (1 - α) · r  −  α · (1 - p)
  where
    α = baseline accuracy at N=1
    p = preserve = P[correct@N | correct@1]
    r = recover  = P[correct@N | wrong@1]

Ensembling helps iff (1-α)r > α(1-p), i.e.,
  r / (1-p) > α / (1-α)            (recovery-to-downgrade odds-ratio condition)

함의:
  - Recovery는 항상 preservation보다 약함 (모든 cell에서 r < p, 보통 r << 1-(1-p))
  - 따라서 baseline α가 클수록 ensembling이 손해를 보기 쉬움
  - 추가로 high-K 도메인일수록 (correlated errors 가설에 따라) recovery가 더 약해질 가능성

비교:
  IID-Condorcet null model: independent agents with single per-agent acc q.
  preserve & recover 모두 q에 의해 결정 → 두 값이 가깝게 움직여야 함.
  실제 데이터에서 preserve >> recover (큰 비대칭) — IID null과 inconsistent.

산출:
  analysis/mechanism_voting_asymmetry.csv         — per-cell α, p, r, condition
  analysis/mechanism_logit_coefs.csv              — item-level logit (auxiliary)
  paper/latex_v2/figures/fig8_mechanism.pdf/.png  — three panels
"""
from __future__ import annotations
import sys
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
FIG = ROOT / 'paper' / 'latex_v2' / 'figures'

matplotlib.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                              'font.family': 'sans-serif',
                              'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})

DOMAIN_COLORS = {'medical': '#d62728', 'legal': '#1f77b4',
                 'logic': '#9467bd', 'long_context': '#ff7f0e'}
MODEL_MARKERS = {'gpt_4o': 'o', 'claude_sonnet_4_6': 's'}


# =====================================================================
# Per-cell preserve / recover from full-cell McNemar tables
# =====================================================================
df_q = pd.read_csv(A / 'per_question_kbenefit.csv')
domain_K = df_q.groupby('domain')['K'].mean().to_dict()
print('Domain-level K (item-level mean):', {k: round(v, 3) for k, v in domain_K.items()})

pair = pd.read_csv(A / 'formal_stats_pairwise.csv')

records = []
for _, r in pair.iterrows():
    if r['a1'] != 1: continue
    K = domain_K.get(r['domain'])
    if K is None: continue
    n = int(r['n_paired'])
    correct_a1 = round(float(r['acc_a1']) * n)
    wrong_a1 = n - correct_a1
    b, c = int(r['mcnemar_b']), int(r['mcnemar_c'])
    p_preserve = (correct_a1 - b) / correct_a1 if correct_a1 else np.nan
    r_recover = c / wrong_a1 if wrong_a1 else np.nan
    alpha = float(r['acc_a1'])
    delta_predicted = (1 - alpha) * r_recover - alpha * (1 - p_preserve)
    delta_observed = float(r['delta_a2_a1'])
    records.append({
        'domain': r['domain'], 'model': r['model'], 'a2': int(r['a2']),
        'K': float(K), 'n_paired': n,
        'alpha_baseline_acc': alpha,
        'preserve': p_preserve, 'recover': r_recover,
        'downgrade_1_minus_p': 1 - p_preserve,
        'delta_observed': delta_observed,
        'delta_predicted_decomp': delta_predicted,
        'helps_iff_recover_gt': alpha / (1 - alpha) * (1 - p_preserve) if (1 - alpha) > 0 else np.nan,
        'condition_satisfied': (
            r_recover > (alpha / (1 - alpha) * (1 - p_preserve)) if (1 - alpha) > 0 else False
        ),
        'mcnemar_b_downgrade': b, 'mcnemar_c_upgrade': c,
        'mcnemar_p': float(r['mcnemar_p']),
    })
cell_df = pd.DataFrame(records)
cell_df.to_csv(A / 'mechanism_voting_asymmetry.csv', index=False)

print('\n=== Per-cell preserve, recover, condition ===')
cols = ['domain', 'model', 'a2', 'K', 'alpha_baseline_acc',
        'preserve', 'recover', 'helps_iff_recover_gt',
        'condition_satisfied', 'delta_observed']
print(cell_df[cols].round(3).to_string(index=False))

# =====================================================================
# Sanity: predicted vs observed Δ should match exactly (algebra check)
# =====================================================================
diff = (cell_df['delta_observed'] - cell_df['delta_predicted_decomp']).abs().max()
print(f'\n[sanity] max |observed − predicted Δ| = {diff:.2e}  (should be ~0)')

# =====================================================================
# Test 1: voting asymmetry — preserve > recover universally?
# =====================================================================
sub2 = cell_df[cell_df['a2'] == 2]
sub4 = cell_df[cell_df['a2'] == 4]
print(f'\n=== Voting asymmetry (preserve > recover) ===')
print(f'  N=2: {(sub2["preserve"] > sub2["recover"]).sum()}/{len(sub2)} cells have p > r')
print(f'  N=4: {(sub4["preserve"] > sub4["recover"]).sum()}/{len(sub4)} cells have p > r')
print(f'  N=2 mean(preserve − recover) = {(sub2["preserve"] - sub2["recover"]).mean():+.3f}')
print(f'  N=4 mean(preserve − recover) = {(sub4["preserve"] - sub4["recover"]).mean():+.3f}')

# =====================================================================
# Test 2: K vs recover (correlated-errors hypothesis)
# =====================================================================
print(f'\n=== Test: does recover decline with K? ===')
for n_val in (2, 4):
    s = cell_df[cell_df['a2'] == n_val].dropna(subset=['recover', 'K'])
    if len(s) < 3: continue
    r_p = stats.pearsonr(s['K'], s['recover'])
    r_s = stats.spearmanr(s['K'], s['recover'])
    print(f'  N={n_val}: Pearson r = {r_p[0]:+.3f} (p={r_p[1]:.3f}),  '
          f'Spearman ρ = {r_s[0]:+.3f} (p={r_s[1]:.3f}),  n={len(s)}')

# =====================================================================
# Item-level logit (auxiliary, smaller-n)
# =====================================================================
import statsmodels.api as sm

print(f'\n=== Item-level logit (n=350) ===')
logit_rows = []
for col_ic, label in [('ic_2', 'N=2'), ('ic_4', 'N=4')]:
    sub = df_q.dropna(subset=['ic_1', col_ic])
    sub_corr = sub[sub['ic_1'] == True]
    sub_wrong = sub[sub['ic_1'] == False]
    for sub2_, name in [(sub_corr, f'preserve_{label}'), (sub_wrong, f'recover_{label}')]:
        try:
            X = sm.add_constant(sub2_['K'].to_numpy())
            res = sm.Logit(sub2_[col_ic].astype(float).to_numpy(), X).fit(disp=0)
            slope, se, p = float(res.params[1]), float(res.bse[1]), float(res.pvalues[1])
            print(f'  {name:18s}: slope_K = {slope:+.3f} ± {se:.3f}, p = {p:.4f}, n = {len(sub2_)}')
            logit_rows.append({'name': name, 'slope_K': slope, 'slope_SE': se,
                                 'p': p, 'n': len(sub2_)})
        except Exception as e:
            print(f'  {name}: fit failed ({e})')

pd.DataFrame(logit_rows).to_csv(A / 'mechanism_logit_coefs.csv', index=False)

# =====================================================================
# Figure
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))


def cell_marker(ax, x, y, dom, model, label_text):
    c = DOMAIN_COLORS[dom]
    m = MODEL_MARKERS[model]
    ax.scatter(x, y, c=c, marker=m, s=200, edgecolor='black', linewidth=1.4, zorder=4)
    ax.annotate(label_text, (x, y), xytext=(7, 5), textcoords='offset points',
                 fontsize=8, color=c)


# (a) preserve vs recover scatter, with diagonal
ax = axes[0]
for _, r in sub2.iterrows():
    if pd.isna(r['preserve']) or pd.isna(r['recover']): continue
    label = f"{r['domain'][:4]}/{'GPT' if r['model']=='gpt_4o' else 'Cl'}"
    cell_marker(ax, r['recover'], r['preserve'], r['domain'], r['model'], label)
ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6, label='preserve = recover')
ax.fill_between([0, 1], [0, 1], [1, 1], color='#1f77b4', alpha=0.06,
                 label='preserve > recover (vote asymmetric)')
ax.set_xlabel('Recovery rate  r = P[correct@2 | wrong@1]')
ax.set_ylabel('Preservation rate  p = P[correct@2 | correct@1]')
ax.set_title('(a) Voting asymmetry:  preserve > recover universally')
ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower right')

# (b) decomposition: observed vs predicted Δ
ax = axes[1]
xx = np.linspace(-0.4, 0.2, 100)
ax.plot(xx, xx, '--', color='gray', alpha=0.6)
for _, r in sub2.iterrows():
    if pd.isna(r['delta_predicted_decomp']): continue
    label = f"{r['domain'][:4]}/{'GPT' if r['model']=='gpt_4o' else 'Cl'}"
    cell_marker(ax, r['delta_predicted_decomp'], r['delta_observed'],
                r['domain'], r['model'], label)
ax.set_xlabel('Predicted Δ from decomposition\n(1−α)·r − α·(1−p)')
ax.set_ylabel('Observed Δ_{N=2 − N=1}')
ax.set_title('(b) Algebraic decomposition is exact')
ax.grid(alpha=0.3); ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)

# (c) condition: r vs threshold α/(1-α) × (1-p)
ax = axes[2]
xx2 = np.linspace(0, 1.5, 100)
ax.plot(xx2, xx2, '--', color='gray', alpha=0.6, label='r = α/(1−α)·(1−p) (break-even)')
ax.fill_between(xx2, xx2, np.maximum(xx2, 1.0), color='#2ca02c', alpha=0.10,
                 label='ensembling helps')
ax.fill_between(xx2, np.zeros_like(xx2), xx2, color='#d62728', alpha=0.10,
                 label='ensembling hurts')
for _, r in sub2.iterrows():
    if pd.isna(r['recover']): continue
    label = f"{r['domain'][:4]}/{'GPT' if r['model']=='gpt_4o' else 'Cl'}"
    cell_marker(ax, r['helps_iff_recover_gt'], r['recover'], r['domain'], r['model'], label)
ax.set_xlim(-0.02, 1.5); ax.set_ylim(-0.02, 0.6)
ax.set_xlabel('α/(1−α) · (1−p)  (downgrade-cost-to-baseline odds)')
ax.set_ylabel('r  (recovery rate)')
ax.set_title('(c) Help-or-hurt decision boundary')
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig.suptitle('Mechanism: voting asymmetry decomposition (cell-level, full data)',
              y=1.04, fontsize=13, fontweight='bold')

out_pdf = FIG / 'fig8_mechanism.pdf'
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(FIG / 'fig8_mechanism.png', bbox_inches='tight')
plt.close()
print(f'\nWrote {out_pdf}')
print(f'Wrote {A / "mechanism_voting_asymmetry.csv"}')
