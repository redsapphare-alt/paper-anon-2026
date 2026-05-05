"""Question-level inter-judge agreement.

각 question에 두 judge가 모두 점수를 매김 (같은 sample 사용 — same seed).
두 judge의 점수를 매칭해서 per-axis 상관관계 계산.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / 'analysis' / 'feature_judge'
A = ROOT / 'analysis'

def load_qlevel(judge: str, domain: str):
    files = sorted(J.glob(f'{judge}_question-level_{domain}_*.json'))
    if not files: return None
    return json.load(open(files[-1], encoding='utf-8'))

def main():
    domains = ['medical', 'legal', 'code', 'logic', 'long_context']
    rows = []
    for dom in domains:
        gpt = load_qlevel('gpt', dom)
        gem = load_qlevel('gemini', dom)
        if not gpt or not gem: continue
        # match by item_id
        gpt_by = {r.get('item_id'): r.get('parsed') for r in gpt if r.get('parsed')}
        gem_by = {r.get('item_id'): r.get('parsed') for r in gem if r.get('parsed')}
        common = set(gpt_by) & set(gem_by)
        for qid in common:
            for axis in ('D','V','K','S'):
                rows.append({'domain': dom, 'item_id': qid, 'axis': axis,
                              'gpt': gpt_by[qid][axis], 'gemini': gem_by[qid][axis]})
    df = pd.DataFrame(rows)
    df.to_csv(A / 'qlevel_per_question_pairs.csv', index=False)

    print('=== Question-level inter-judge correlation (per axis, across all domains) ===')
    print(f'{"axis":<6}{"n":>6}{"pearson r":>13}{"p":>10}{"MAE":>8}')
    for axis in ('D','V','K','S'):
        sub = df[df['axis'] == axis]
        if len(sub) < 5: continue
        r, p = stats.pearsonr(sub['gpt'], sub['gemini'])
        mae = (sub['gpt'] - sub['gemini']).abs().mean()
        print(f'{axis:<6}{len(sub):>6}{r:>13.3f}{p:>10.2e}{mae:>8.3f}')

    print('\n=== Per (domain, axis) correlation ===')
    print(f'{"domain":<14}{"axis":<6}{"n":>6}{"r":>10}{"p":>10}{"MAE":>8}')
    for dom in domains:
        for axis in ('D','V','K','S'):
            sub = df[(df['domain'] == dom) & (df['axis'] == axis)]
            if len(sub) < 5: continue
            try:
                r, p = stats.pearsonr(sub['gpt'], sub['gemini'])
            except Exception:
                r, p = float('nan'), float('nan')
            mae = (sub['gpt'] - sub['gemini']).abs().mean()
            print(f'{dom:<14}{axis:<6}{len(sub):>6}{r:>10.3f}{p:>10.2e}{mae:>8.3f}')

    # ICC-like: pool gpt-gem pairs and compute (mean_gpt, mean_gem) agreement at domain×axis level
    print('\n=== Domain × axis means (Q-level both judges) ===')
    pivot = df.groupby(['domain','axis']).agg(
        gpt_mean=('gpt','mean'), gpt_std=('gpt','std'),
        gem_mean=('gemini','mean'), gem_std=('gemini','std'),
        mae=('gpt', lambda s: (df.loc[s.index, 'gpt'] - df.loc[s.index, 'gemini']).abs().mean()),
    ).reset_index()
    print(pivot.round(3).to_string(index=False))
    pivot.to_csv(A / 'qlevel_domain_axis_means.csv', index=False)

if __name__ == '__main__':
    main()
