"""K 임계값 τ에 대한 sensitivity analysis.

τ=0.6은 hypothesis-driven choice. 이 선택이 cherry-picked가 아니라는 걸 보이려면,
- τ를 0.3~0.9 범위에서 sweep
- 각 τ에서 N* 예측 정확도 계산
- 결정 규칙이 어느 τ 범위에서 강건한지 시각화

세 가지 평가 metric:
  (1) Cell-level (8 trustable cells): exact match for N*
  (2) Per-question (n=350): "predicted vs observed agent benefit" agreement
  (3) Bootstrap CI on (1)

산출:
  analysis/K_sensitivity.csv
  analysis/K_sensitivity.json
  paper/latex_v2/figures/fig6_K_sensitivity.pdf
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
FIG = ROOT / 'paper' / 'latex_v2' / 'figures'

matplotlib.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
})

# ===== Cell-level data =====
phase3_df = pd.read_csv(A / 'phase3_dataset.csv')
print(f'Cell-level dataset: {len(phase3_df)} rows')
print(phase3_df[['domain','model','K','peak']].to_string())

# ===== Per-question data =====
pq_df = pd.read_csv(A / 'per_question_kbenefit.csv')
pq_df = pq_df.dropna(subset=['benefit_2_1'])
print(f'\nPer-question dataset: {len(pq_df)} rows')

# ===== Sweep τ =====
TAU_GRID = np.arange(0.30, 0.96, 0.02)

cell_metrics = []
pq_metrics = []

# observed peak from phase3 (manually map: medical/Claude excluded due to contamination)
# But we use the dataset as-is from phase3
for tau in TAU_GRID:
    # Cell-level: predict N*=1 if K>=tau else N*=2
    pred = np.where(phase3_df['K'] >= tau, 1, 2)
    truth = phase3_df['peak'].astype(int).to_numpy()
    exact = float((pred == truth).mean())
    mae = float(np.abs(pred - truth).mean())
    cell_metrics.append({'tau': tau, 'exact_match': exact, 'mae': mae,
                          'n': len(phase3_df)})

    # Per-question: predict that going N=1 → N=2 will HARM if K>=tau (else benefit)
    # observed: benefit_2_1 sign
    pred_benefit = np.where(pq_df['K'] >= tau, -1, 1)  # +1 = expect benefit
    obs_sign = np.sign(pq_df['benefit_2_1']).astype(int).to_numpy()
    # agreement: predicted matches sign (counting 0 separately)
    matches = (pred_benefit == obs_sign).mean()
    # also: P(harm) when high-K vs low-K (manual contingency)
    high_K = pq_df['K'] >= tau
    n_high_harm = int((high_K & (pq_df['benefit_2_1'] < 0)).sum())
    n_high_total = int(high_K.sum())
    n_low_imp = int((~high_K & (pq_df['benefit_2_1'] > 0)).sum())
    n_low_total = int((~high_K).sum())
    p_harm_high = n_high_harm / n_high_total if n_high_total else 0
    p_imp_low = n_low_imp / n_low_total if n_low_total else 0
    pq_metrics.append({
        'tau': tau, 'sign_agreement': float(matches),
        'p_harm_given_highK': p_harm_high,
        'p_improve_given_lowK': p_imp_low,
        'n_high': n_high_total, 'n_low': n_low_total,
    })

cell_df = pd.DataFrame(cell_metrics)
pq_metrics_df = pd.DataFrame(pq_metrics)
print('\n=== Cell-level sensitivity ===')
print(cell_df.round(3).to_string(index=False))
print('\n=== Per-question sensitivity ===')
print(pq_metrics_df.round(3).to_string(index=False))

# ===== Bootstrap CI for cell-level exact match =====
# at τ=0.6 (and a few other τ values)
rng = np.random.default_rng(42)
N_BOOT = 5000
tau_anchors = [0.40, 0.50, 0.60, 0.70, 0.80]
boot_cis = {}
for tau in tau_anchors:
    pred = np.where(phase3_df['K'] >= tau, 1, 2)
    truth = phase3_df['peak'].astype(int).to_numpy()
    n = len(truth)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots.append(float((pred[idx] == truth[idx]).mean()))
    boot_cis[tau] = (float(np.mean(boots)),
                      float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5)))

print(f'\n=== Bootstrap 95% CI for cell-level exact match (n_boot={N_BOOT}) ===')
for tau, (mean, lo, hi) in boot_cis.items():
    print(f'  τ={tau:.2f}: mean={mean:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]')

# ===== Save =====
cell_df.to_csv(A / 'K_sensitivity_cell.csv', index=False)
pq_metrics_df.to_csv(A / 'K_sensitivity_pq.csv', index=False)
summary = {
    'tau_grid': TAU_GRID.tolist(),
    'cell_level': cell_metrics,
    'per_question': pq_metrics,
    'bootstrap_cell_CI': {str(t): {'mean': v[0], 'ci_lo': v[1], 'ci_hi': v[2]}
                          for t, v in boot_cis.items()},
    'best_tau_cell': float(cell_df.loc[cell_df['exact_match'].idxmax(), 'tau']),
    'best_tau_cell_max_exact': float(cell_df['exact_match'].max()),
    'plateau_range': [float(cell_df.loc[cell_df['exact_match'] >= cell_df['exact_match'].max() - 1e-6, 'tau'].min()),
                      float(cell_df.loc[cell_df['exact_match'] >= cell_df['exact_match'].max() - 1e-6, 'tau'].max())],
}
(A / 'K_sensitivity.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'\nMax cell-level exact at τ ∈ [{summary["plateau_range"][0]:.2f}, {summary["plateau_range"][1]:.2f}]')

# ===== Figure =====
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) Cell-level
ax = axes[0]
ax.plot(cell_df['tau'], cell_df['exact_match'], '-o', color='#1f77b4', lw=2, markersize=4,
         label='Cell-level (n=8)')
# shaded plateau
plat_lo = cell_df.loc[cell_df['exact_match'] >= cell_df['exact_match'].max() - 1e-6, 'tau'].min()
plat_hi = cell_df.loc[cell_df['exact_match'] >= cell_df['exact_match'].max() - 1e-6, 'tau'].max()
ax.axvspan(plat_lo, plat_hi, color='#1f77b4', alpha=0.12,
           label=f'Plateau τ ∈ [{plat_lo:.2f}, {plat_hi:.2f}]')
ax.axvline(0.6, color='purple', linestyle=':', label='Hypothesised τ = 0.6')
# overlay bootstrap CIs at anchor τ
for tau, (mean, lo, hi) in boot_cis.items():
    ax.errorbar([tau], [mean], yerr=[[mean - lo], [hi - mean]],
                 fmt='s', color='#d62728', capsize=4, markersize=8, alpha=0.85, zorder=4)
ax.set_xlabel('K threshold τ')
ax.set_ylabel('Cell-level exact match (N* prediction)')
ax.set_title('(a) Cell-level: prediction is robust over τ ∈ [{:.2f}, {:.2f}]'.format(plat_lo, plat_hi))
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

# (b) Per-question
ax = axes[1]
ax.plot(pq_metrics_df['tau'], pq_metrics_df['sign_agreement'], '-', color='#2ca02c', lw=2, label='Sign agreement (n=350)')
ax.plot(pq_metrics_df['tau'], pq_metrics_df['p_harm_given_highK'], '-', color='#d62728', lw=2, label='P(harm | K ≥ τ)')
ax.plot(pq_metrics_df['tau'], pq_metrics_df['p_improve_given_lowK'], '-', color='#1f77b4', lw=2, label='P(improve | K < τ)')
ax.axvline(0.6, color='purple', linestyle=':', alpha=0.6)
ax.set_xlabel('K threshold τ')
ax.set_ylabel('Probability')
ax.set_title('(b) Per-question: monotone signal in K')
ax.set_ylim(0, 1.0)
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.suptitle('K threshold τ sensitivity', y=1.02, fontsize=12, fontweight='bold')
out = FIG / 'fig6_K_sensitivity.pdf'
plt.savefig(out, bbox_inches='tight')
plt.savefig(FIG / 'fig6_K_sensitivity.png', bbox_inches='tight')
plt.close()
print(f'\nWrote {out}')
