"""실험 결과의 정답 판정 sanity check.

각 도메인에서:
1. predicted_answer 분포 (빈값/refusal/형식 이상)
2. is_correct 판정 로직 점검 - 빈 답변/refusal에 대해 어떻게 처리되는지
3. ground_truth와 predicted_answer 매칭 방식 추론
4. 의심스러운 케이스 샘플
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

# UTF-8 출력 (한국어/특수문자 안전)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'results' / 'merged'

REFUSAL_PATTERNS = [
    r"i'?m sorry",
    r"i cannot",
    r"i can'?t",
    r"unable to",
    r"i don'?t (know|have)",
    r"as an ai",
    r"^sorry,",
]
REFUSAL_RE = re.compile('|'.join(REFUSAL_PATTERNS), re.IGNORECASE)

def short(s, n=200):
    if not isinstance(s, str): return repr(s)
    s = s.replace('\n', ' / ')
    return s if len(s) <= n else s[:n] + '...[+]'

def analyze_file(path: Path):
    d = json.load(open(path, encoding='utf-8'))
    res = d.get('results', [])
    n = len(res)
    if n == 0:
        return
    print(f"\n{'='*80}\n{path.name}  (n={n})\n{'='*80}")

    # quick distributions
    empty_pred = 0
    refusal_in_pred = 0
    refusal_in_reasoning = 0
    is_correct_when_empty_pred = 0
    is_correct_when_refusal_in_reasoning = 0
    pred_len_dist = []
    gt_len_dist = []
    gt_values = []  # for legal/medical/logic — multiple choice letters?

    for r in res:
        pa = r.get('predicted_answer') or ''
        gt = r.get('ground_truth') or ''
        rs = r.get('reasoning') or ''
        ic = bool(r.get('is_correct'))

        pred_len_dist.append(len(pa.strip()))
        gt_len_dist.append(len(gt.strip()))
        gt_values.append(gt.strip())

        if not pa.strip():
            empty_pred += 1
            if ic:
                is_correct_when_empty_pred += 1
        if REFUSAL_RE.search(pa):
            refusal_in_pred += 1
        if REFUSAL_RE.search(rs):
            refusal_in_reasoning += 1
            if ic:
                is_correct_when_refusal_in_reasoning += 1

    print(f"empty predicted_answer:       {empty_pred:5d}/{n}  ({empty_pred/n*100:.2f}%)")
    print(f"  └ marked is_correct=True:   {is_correct_when_empty_pred} 🚨" if is_correct_when_empty_pred else
          f"  └ marked is_correct=True:   {is_correct_when_empty_pred}")
    print(f"refusal-pattern in predicted: {refusal_in_pred:5d}/{n}  ({refusal_in_pred/n*100:.2f}%)")
    print(f"refusal-pattern in reasoning: {refusal_in_reasoning:5d}/{n}  ({refusal_in_reasoning/n*100:.2f}%)")
    print(f"  └ but is_correct=True:      {is_correct_when_refusal_in_reasoning} 🚨" if is_correct_when_refusal_in_reasoning else
          f"  └ but is_correct=True:      {is_correct_when_refusal_in_reasoning}")

    # length stats
    import statistics as st
    print(f"predicted_answer length:  median={st.median(pred_len_dist):.0f}  "
          f"mean={st.mean(pred_len_dist):.0f}  max={max(pred_len_dist)}")
    print(f"ground_truth length:      median={st.median(gt_len_dist):.0f}  "
          f"mean={st.mean(gt_len_dist):.0f}  max={max(gt_len_dist)}")

    # GT value distribution (if short — likely MCQ letter)
    short_gts = [g for g in gt_values if len(g) <= 5]
    if len(short_gts) > n * 0.5:
        from collections import Counter
        common = Counter(short_gts).most_common(8)
        print(f"GT value top-8 (short):   {common}")

    # 3 suspicious samples: empty predicted but is_correct=True
    samples_emp_correct = [r for r in res if not (r.get('predicted_answer') or '').strip() and r.get('is_correct')]
    if samples_emp_correct:
        print(f"\n  ⚠️  {len(samples_emp_correct)} cases: predicted_answer='' but is_correct=True")
        for r in samples_emp_correct[:2]:
            print(f"    qid={r.get('question_id')}  gt={r.get('ground_truth')!r}  reasoning={short(r.get('reasoning'), 200)!r}")

    # 3 suspicious samples: refusal in reasoning but is_correct=True
    samples_ref_correct = [r for r in res
                           if REFUSAL_RE.search(r.get('reasoning') or '') and r.get('is_correct')]
    if samples_ref_correct:
        print(f"\n  ⚠️  {len(samples_ref_correct)} cases: refusal in reasoning but is_correct=True")
        for r in samples_ref_correct[:2]:
            print(f"    qid={r.get('question_id')}  gt={r.get('ground_truth')!r}  pred={short(r.get('predicted_answer'), 100)!r}  reasoning={short(r.get('reasoning'), 200)!r}")

    # show 2 representative correct + 2 wrong for matching pattern
    print("\n  representative samples:")
    correct_samples = [r for r in res if r.get('is_correct')][:2]
    wrong_samples = [r for r in res if not r.get('is_correct')][:2]
    for label, items in (('CORRECT', correct_samples), ('WRONG  ', wrong_samples)):
        for r in items:
            pa = short(r.get('predicted_answer'), 120)
            gt = short(r.get('ground_truth'), 80)
            print(f"    [{label}] gt={gt!r:40s}  pred={pa!r}")


def main():
    targets = {
        'medical': 'medical_gpt_4o_1agent.json',
        'legal':   'legal_gpt_4o_1agent_parallel.json',
        'code_gpt':    'code_gpt_4o_1agent.json',
        'code_claude': 'code_claude_sonnet_4_6_4agent.json',  # 1% only — suspicious
        'logic':   'logic_gpt_4o_3agent.json',
        'long_context_gpt': 'long_context_gpt_4o_2agent_parallel.json',
        'long_context_claude': 'long_context_claude_sonnet_4_6_2agent.json',
    }
    for label, fname in targets.items():
        p = M / fname
        if not p.exists():
            print(f"MISSING: {fname}")
            continue
        analyze_file(p)

if __name__ == '__main__':
    main()
