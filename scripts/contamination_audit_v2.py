"""정답 판정 오염도 재점검 (v2 — 스키마 차이 해결).

발견된 스키마 차이:
- GPT 파일: ground_truth + predicted_answer
- Claude 파일: correct_answer + predicted_answer
- (둘 다 가질 수도 있음 — 우선순위로 ground_truth 먼저, 없으면 correct_answer)

진짜 버그:
1. predicted_answer = '' 인데 is_correct=True (빈 답변 정답 처리)
2. predicted_answer 길이가 200자에 캡 (truncation) — 매칭 로직에 따라 영향 다름
3. reasoning에 refusal pattern인데 is_correct=True
4. predicted_answer 형식 이상 (e.g., 코드 도메인에서 추출 실패)
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from collections import Counter
import pandas as pd

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'results' / 'merged'
OUT = ROOT / 'analysis'

REFUSAL = re.compile(
    r"i'?m sorry|i cannot|i can'?t|unable to|i don'?t (know|have)|"
    r"as an ai|need (more |the )?(context|options|choices|question|information)",
    re.IGNORECASE,
)
DOMAIN_SIZES = {'medical': 5000, 'legal': 3000, 'code': 1000, 'logic': 1500, 'long_context': 902}

def parse_name(name: str):
    base = name.replace('.json', '')
    parts = base.split('_')
    a_idx = next((i for i, p in enumerate(parts) if p.endswith('agent') and p[:-5].isdigit()), None)
    if a_idx is None: return None
    agents = int(parts[a_idx][:-5])
    for dom in sorted(DOMAIN_SIZES, key=lambda x: -x.count('_')):
        prefix = dom.split('_')
        if parts[:len(prefix)] == prefix:
            model = '_'.join(parts[len(prefix):a_idx])
            return dom, model, agents
    return None

def get_gt(r):
    return (r.get('ground_truth') or r.get('correct_answer') or '').strip()

rows = []
for f in sorted(M.glob('*.json')):
    parsed = parse_name(f.name)
    if not parsed: continue
    domain, model, agents = parsed
    try:
        d = json.load(open(f, encoding='utf-8'))
    except Exception as e:
        print(f"ERR {f.name}: {e}"); continue
    res = d.get('results', [])
    n = len(res)
    if n == 0: continue

    # schema check
    has_gt = sum(1 for r in res if r.get('ground_truth'))
    has_ca = sum(1 for r in res if r.get('correct_answer'))
    schema = 'ground_truth' if has_gt > has_ca else ('correct_answer' if has_ca else 'NEITHER')

    empty_pred = empty_pred_correct = empty_gt = refusal_in_pred = refusal_correct = trunc = correct_total = 0
    short_pred_correct_pct = 0  # 짧은 pred (<=2자) 중 정답률
    short_pred_count = 0
    short_pred_correct = 0
    pred_lens = []

    for r in res:
        pa = (r.get('predicted_answer') or '').strip()
        gt = get_gt(r)
        rs = r.get('reasoning') or ''  # may not exist for Claude
        ic = bool(r.get('is_correct'))
        pred_lens.append(len(pa))

        if ic: correct_total += 1
        if not pa:
            empty_pred += 1
            if ic: empty_pred_correct += 1
        if not gt:
            empty_gt += 1
        if REFUSAL.search(pa) or REFUSAL.search(rs):
            refusal_in_pred += 1
            if ic: refusal_correct += 1
        if 199 <= len(r.get('predicted_answer') or '') <= 200:
            trunc += 1
        if 0 < len(pa) <= 2:
            short_pred_count += 1
            if ic: short_pred_correct += 1

    reported_acc = correct_total / n * 100
    corrected = (correct_total - empty_pred_correct) / n * 100
    inflation = reported_acc - corrected

    rows.append({
        'domain': domain, 'model': model, 'agents': agents,
        'n': n, 'schema': schema,
        'empty_gt': empty_gt,
        'empty_pred': empty_pred,
        'empty_pred_BUG_correct': empty_pred_correct,
        'refusal_BUG_correct': refusal_correct,
        'pred_truncated_200': trunc,
        'pred_truncated_pct': trunc / n * 100,
        'short_pred_correct_rate': (short_pred_correct / max(short_pred_count, 1)) * 100,
        'reported_correct': correct_total,
        'reported_accuracy': reported_acc,
        'corrected_acc_minus_empty': corrected,
        'inflation_pp': inflation,
        'file': f.name,
    })

df = pd.DataFrame(rows).sort_values(['domain', 'model', 'agents'])

# === Output ===
print('=== Schema check ===')
print(df.groupby(['domain', 'model', 'schema']).size().to_string())
print()

# main per-cell view
print('=== Per-cell audit (sorted by inflation) ===')
view = df[['domain', 'model', 'agents', 'n', 'schema', 'empty_pred',
           'empty_pred_BUG_correct', 'refusal_BUG_correct', 'pred_truncated_200',
           'reported_accuracy', 'corrected_acc_minus_empty', 'inflation_pp']].copy()
view = view.sort_values('inflation_pp', ascending=False)
print(view.head(20).round(2).to_string(index=False))
print()

# Critical
print('=== Critical: empty-pred-but-is_correct=True bug ===')
buggy = df[df['empty_pred_BUG_correct'] > 0].sort_values('empty_pred_BUG_correct', ascending=False)
print(f"Total cells affected: {len(buggy)} / {len(df)}")
for _, r in buggy.head(15).iterrows():
    pct_of_correct = r['empty_pred_BUG_correct'] / max(r['reported_correct'], 1) * 100
    print(f"  {r['domain']:14s} {r['model']:20s} {r['agents']}-agent: "
          f"{r['empty_pred_BUG_correct']}/{r['reported_correct']} 'correct' are EMPTY pred "
          f"({pct_of_correct:.1f}%)  →  reported {r['reported_accuracy']:.2f}% "
          f"corrected {r['corrected_acc_minus_empty']:.2f}%")
print()

print('=== Truncation at 200 chars (predicted_answer cap?) ===')
trunc = df[df['pred_truncated_pct'] > 30].sort_values('pred_truncated_pct', ascending=False)
print(f"Cells with >30% truncated: {len(trunc)}")
for _, r in trunc.head(15).iterrows():
    print(f"  {r['domain']:14s} {r['model']:20s} {r['agents']}-agent: "
          f"{r['pred_truncated_200']}/{r['n']} truncated ({r['pred_truncated_pct']:.1f}%)  "
          f"acc={r['reported_accuracy']:.2f}%")
print()

print('=== Short-prediction (1-2 char) accuracy ===')
print(df[['domain','model','agents','schema','short_pred_correct_rate','reported_accuracy']]
      .sort_values('short_pred_correct_rate', ascending=False).head(15).round(2).to_string(index=False))

df.to_csv(OUT / 'contamination_audit_v2.csv', index=False, encoding='utf-8')
print(f"\nWrote {OUT / 'contamination_audit_v2.csv'}")
