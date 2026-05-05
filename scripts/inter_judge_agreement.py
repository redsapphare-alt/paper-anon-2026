"""두 judge간 도메인-레벨 점수 일치도 평가.

산출:
  analysis/judge_agreement.csv  — judge별 도메인×axis 점수 + delta
  분포 + correlation 콘솔 출력
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'analysis'

d = json.load(open(A / 'domain_features.json', encoding='utf-8'))
agg = d['aggregated']
raw = d['raw_per_domain_judge']

# Build long table
rows = []
for dom, by_judge in raw.items():
    for judge, reps in by_judge.items():
        for r in reps:
            p = r.get('parsed')
            if not p: continue
            for axis in ('D', 'V', 'K', 'S'):
                rows.append({'domain': dom, 'judge': judge, 'rep': r.get('rep', '?'),
                              'axis': axis, 'value': p[axis]})

df = pd.DataFrame(rows)

# Per (domain, judge) means
mean_by_dj = df.groupby(['domain', 'judge', 'axis'])['value'].mean().unstack('axis').reset_index()
mean_by_dj.to_csv(A / 'judge_agreement.csv', index=False)
print('=== Judge means per (domain, axis) ===')
print(mean_by_dj.round(3).to_string(index=False))
print()

# Pivot for inter-judge analysis
# Per axis, get gpt vs gemini means across 5 domains
print('=== Per-axis Pearson correlation between judges (across 5 domains) ===')
for axis in ('D', 'V', 'K', 'S'):
    sub = mean_by_dj.pivot(index='domain', columns='judge', values=axis)
    if 'gpt' in sub.columns and 'gemini' in sub.columns:
        sub = sub.dropna()
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub['gpt'].to_numpy(), sub['gemini'].to_numpy())
            mae = float((sub['gpt'] - sub['gemini']).abs().mean())
            print(f"  {axis}: pearson r={r:+.3f} (p={p:.3f})  MAE={mae:.3f}")
        else:
            print(f"  {axis}: insufficient data")

# Overall inter-judge: stack all (domain, axis) points and compute corr
gpt_pts = []
gem_pts = []
for axis in ('D', 'V', 'K', 'S'):
    sub = mean_by_dj.pivot(index='domain', columns='judge', values=axis).dropna()
    if 'gpt' in sub.columns and 'gemini' in sub.columns:
        gpt_pts += sub['gpt'].tolist()
        gem_pts += sub['gemini'].tolist()
if len(gpt_pts) >= 3:
    r, p = stats.pearsonr(gpt_pts, gem_pts)
    mae = sum(abs(a-b) for a,b in zip(gpt_pts, gem_pts)) / len(gpt_pts)
    print(f"\nOVERALL inter-judge (5 domains × 4 axes = {len(gpt_pts)} points):")
    print(f"  Pearson r = {r:+.3f} (p={p:.3e})")
    print(f"  MAE = {mae:.3f}")

# Within-judge consistency (mean std across reps per (domain, axis))
print('\n=== Within-judge consistency (std across 3 reps) ===')
within = df.groupby(['judge', 'domain', 'axis'])['value'].std().reset_index()
print(within.pivot_table(index='judge', values='value', aggfunc='mean').round(4).to_string())
print()

# Combined judge final scores (average of gpt + gemini means)
print('=== Combined final scores (mean across judges) ===')
combined_rows = []
for dom in mean_by_dj['domain'].unique():
    sub = mean_by_dj[mean_by_dj['domain'] == dom]
    row = {'domain': dom}
    for axis in ('D', 'V', 'K', 'S'):
        vals = sub[axis].dropna().tolist()
        if vals:
            row[axis] = sum(vals)/len(vals)
            row[f'{axis}_judge_diff'] = max(vals) - min(vals)
    combined_rows.append(row)
combined_df = pd.DataFrame(combined_rows)
print(combined_df.round(3).to_string(index=False))
combined_df.to_csv(A / 'domain_features_combined.csv', index=False)
print(f"\nWrote {A / 'domain_features_combined.csv'}")
print(f"Wrote {A / 'judge_agreement.csv'}")
