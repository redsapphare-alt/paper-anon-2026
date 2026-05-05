"""Question-level LLM judge 결과 집계.

도메인별·judge별로 (D, V, K, S) 분포 (mean, std, percentiles) 계산.
Domain-level 결과와 비교해 within-domain heterogeneity 평가.

출력:
  analysis/question_features_summary.csv
  analysis/question_features_distribution.json
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
import pandas as pd

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / 'analysis' / 'feature_judge'
A = ROOT / 'analysis'

def load_qlevel(judge: str, domain: str):
    files = sorted(J.glob(f'{judge}_question-level_{domain}_*.json'))
    if not files: return None
    # take latest
    return json.load(open(files[-1], encoding='utf-8'))

def main():
    domains = ['medical', 'legal', 'code', 'logic', 'long_context']
    judges = ['gpt', 'gemini']

    rows = []
    distrib = {}
    for dom in domains:
        distrib[dom] = {}
        for j in judges:
            data = load_qlevel(j, dom)
            if not data: continue
            parsed = [r['parsed'] for r in data if r.get('parsed')]
            if not parsed: continue
            stats_per_axis = {}
            for axis in ('D', 'V', 'K', 'S'):
                vals = [r[axis] for r in parsed]
                stats_per_axis[axis] = {
                    'mean': st.mean(vals),
                    'std': st.stdev(vals) if len(vals) > 1 else 0,
                    'min': min(vals), 'max': max(vals),
                    'p25': sorted(vals)[len(vals)//4],
                    'p50': sorted(vals)[len(vals)//2],
                    'p75': sorted(vals)[3*len(vals)//4],
                    'n': len(vals),
                }
            distrib[dom][j] = stats_per_axis
            for axis in ('D', 'V', 'K', 'S'):
                rows.append({
                    'domain': dom, 'judge': j, 'axis': axis,
                    'mean': stats_per_axis[axis]['mean'],
                    'std': stats_per_axis[axis]['std'],
                    'min': stats_per_axis[axis]['min'],
                    'max': stats_per_axis[axis]['max'],
                    'p50': stats_per_axis[axis]['p50'],
                    'n': stats_per_axis[axis]['n'],
                })
    df = pd.DataFrame(rows)
    df.to_csv(A / 'question_features_summary.csv', index=False)
    (A / 'question_features_distribution.json').write_text(
        json.dumps(distrib, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=== Question-level features (mean ± std across N questions per cell) ===\n')
    pivot = df.pivot_table(index=['domain', 'axis'], columns='judge', values=['mean', 'std'])
    print(pivot.round(3).to_string())
    print()

    # Domain-level (D-level) vs question-level (Q-level) comparison
    feat_d = pd.read_csv(A / 'domain_features_combined.csv')
    print('=== Domain-level vs Question-level comparison (means) ===')
    print(f'{"domain":<14}{"axis":<5}{"D-level":>10}{"Q-level (gpt)":>15}{"Q-level (gem)":>15}{"Q-std (gpt)":>13}')
    print('-'*72)
    def fmt(v, w):
        if v is None: return f'{"-":>{w}}'
        return f'{v:>{w}.2f}'
    for dom in domains:
        if dom not in distrib: continue
        for axis in ('D', 'V', 'K', 'S'):
            dl = float(feat_d[feat_d['domain'] == dom][axis].iloc[0]) if axis in feat_d.columns else None
            ql_gpt = distrib[dom].get('gpt', {}).get(axis, {}).get('mean')
            ql_gem = distrib[dom].get('gemini', {}).get(axis, {}).get('mean')
            ql_gpt_std = distrib[dom].get('gpt', {}).get(axis, {}).get('std')
            print(f'{dom:<14}{axis:<5}'
                  f'{fmt(dl, 10)}{fmt(ql_gpt, 15)}{fmt(ql_gem, 15)}{fmt(ql_gpt_std, 13)}')

    print(f'\nWrote {A / "question_features_summary.csv"}')
    print(f'Wrote {A / "question_features_distribution.json"}')

if __name__ == '__main__':
    main()
