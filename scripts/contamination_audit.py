"""모든 결과 파일에 대해 정답 판정 오염도(contamination)를 정량화.

탐지하는 문제:
1. predicted_answer = '' 인데 is_correct=True (빈 답변이 정답 처리됨)
2. ground_truth = '' 가 다수 — GT 추출 실패 (Claude code/long_context 등)
3. predicted_answer 길이가 200자에 묶여 있음 (truncation)
4. reasoning에 refusal pattern인데 is_correct=True

각 (domain, model, agents)에 대해:
- 원래 보고된 correct 수
- 그 중 empty-pred-correct 수 (false positive)
- "정상" correct 수 (보정 후)
- 정확도 인플레이션 양
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'results' / 'merged'
OUT = ROOT / 'analysis'

REFUSAL = re.compile(r"i'?m sorry|i cannot|i can'?t|unable to|i don'?t (know|have)|as an ai", re.IGNORECASE)
TRUNC_LEN = 200  # observed cap

DOMAIN_SIZES = {'medical': 5000, 'legal': 3000, 'code': 1000, 'logic': 1500, 'long_context': 902}

rows = []

for f in sorted(M.glob('*.json')):
    name = f.name
    # parse domain/model/agents
    # patterns: {domain}_{model}_{N}agent[_parallel|_subprocess|_backup_*].json
    base = name.replace('.json', '')
    parts = base.split('_')
    # find agent token
    a_idx = None
    for i, p in enumerate(parts):
        if p.endswith('agent') and p[:-5].isdigit():
            a_idx = i; break
    if a_idx is None:
        continue
    agents = int(parts[a_idx][:-5])
    # domain = first 1-2 tokens, model = the rest before agent
    # try matching DOMAIN_SIZES keys greedily
    for dom in sorted(DOMAIN_SIZES.keys(), key=lambda x: -x.count('_')):
        prefix = dom.split('_')
        if parts[:len(prefix)] == prefix:
            domain = dom
            model = '_'.join(parts[len(prefix):a_idx])
            break
    else:
        continue

    try:
        d = json.load(open(f, encoding='utf-8'))
    except Exception as e:
        print(f"ERR loading {name}: {e}")
        continue
    res = d.get('results', [])
    n = len(res)
    if n == 0:
        continue

    empty_pred = 0
    empty_pred_correct = 0
    empty_gt = 0
    refusal_reason_correct = 0
    truncated_pred = 0  # exactly 200 chars (or 199-200 boundary)
    correct_total = 0

    for r in res:
        pa = (r.get('predicted_answer') or '').strip()
        gt = (r.get('ground_truth') or '').strip()
        rs = r.get('reasoning') or ''
        ic = bool(r.get('is_correct'))

        if ic: correct_total += 1
        if not pa:
            empty_pred += 1
            if ic: empty_pred_correct += 1
        if not gt:
            empty_gt += 1
        if REFUSAL.search(rs) and ic:
            refusal_reason_correct += 1
        if 199 <= len(r.get('predicted_answer') or '') <= 200:
            truncated_pred += 1

    reported_acc = correct_total / n * 100
    corrected_correct = correct_total - empty_pred_correct
    corrected_acc = corrected_correct / n * 100
    inflation_pp = reported_acc - corrected_acc

    rows.append({
        'domain': domain, 'model': model, 'agents': agents,
        'n_results': n,
        'empty_gt': empty_gt,
        'empty_gt_pct': empty_gt / n * 100,
        'empty_pred': empty_pred,
        'empty_pred_correct (BUG)': empty_pred_correct,
        'refusal_correct (BUG)': refusal_reason_correct,
        'pred_truncated_200': truncated_pred,
        'pred_truncated_pct': truncated_pred / n * 100,
        'reported_correct': correct_total,
        'reported_accuracy': reported_acc,
        'corrected_accuracy_minus_empty': corrected_acc,
        'inflation_pp': inflation_pp,
        'file': name,
    })

df = pd.DataFrame(rows).sort_values(['domain', 'model', 'agents'])

# Summary printout
print('=== Per-cell contamination audit ===')
cols = ['domain', 'model', 'agents', 'n_results', 'empty_gt', 'empty_pred',
        'empty_pred_correct (BUG)', 'pred_truncated_200',
        'reported_accuracy', 'corrected_accuracy_minus_empty', 'inflation_pp']
print(df[cols].round(2).to_string(index=False))
print()

# headline metrics
print('=== Critical findings ===')
buggy_empty = df[df['empty_pred_correct (BUG)'] > 0]
print(f"\n1) Cells with empty-pred-but-correct=True bug: {len(buggy_empty)} / {len(df)} files")
if len(buggy_empty):
    print('   Worst offenders (>50 cases):')
    for _, r in buggy_empty.sort_values('empty_pred_correct (BUG)', ascending=False).head(10).iterrows():
        print(f"   - {r['domain']:14s} {r['model']:20s} {r['agents']}-agent: "
              f"{r['empty_pred_correct (BUG)']}/{r['reported_correct']} 'correct' are empty "
              f"({r['empty_pred_correct (BUG)']/max(r['reported_correct'],1)*100:.1f}% of reported correct)")

empty_gt = df[df['empty_gt_pct'] > 50]
print(f"\n2) Cells with >50% empty ground_truth (extraction broken): {len(empty_gt)}")
for _, r in empty_gt.iterrows():
    print(f"   - {r['domain']:14s} {r['model']:20s} {r['agents']}-agent: "
          f"empty_gt={r['empty_gt']}/{r['n_results']} ({r['empty_gt_pct']:.1f}%)")

trunc = df[df['pred_truncated_pct'] > 30]
print(f"\n3) Cells with >30% predictions truncated at 200 chars: {len(trunc)}")
for _, r in trunc.iterrows():
    print(f"   - {r['domain']:14s} {r['model']:20s} {r['agents']}-agent: "
          f"truncated={r['pred_truncated_200']}/{r['n_results']} ({r['pred_truncated_pct']:.1f}%)")

big_inflation = df[df['inflation_pp'] > 1]
print(f"\n4) Cells with >1pp accuracy inflation from empty-pred bug: {len(big_inflation)}")
for _, r in big_inflation.sort_values('inflation_pp', ascending=False).head(10).iterrows():
    print(f"   - {r['domain']:14s} {r['model']:20s} {r['agents']}-agent: "
          f"reported={r['reported_accuracy']:.2f}% → corrected={r['corrected_accuracy_minus_empty']:.2f}% "
          f"(Δ={r['inflation_pp']:.2f}pp)")

df.to_csv(OUT / 'contamination_audit.csv', index=False, encoding='utf-8')
print(f"\nWrote {OUT / 'contamination_audit.csv'}")
