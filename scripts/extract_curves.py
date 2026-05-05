"""results/merged/*.json에서 (domain, model, agent_count) → 정확도 표를 추출.

per-question 정답 여부도 보존해서 paired test, bootstrap CI에 사용.
출력:
  analysis/accuracy_table.csv         — 한 행 = (domain, model, agent_count) + accuracy/correct/total
  analysis/per_question.parquet       — 한 행 = 한 질문의 정답 여부 (long form)
  analysis/per_question.csv           — 동일하지만 csv (parquet 없을 때 대비)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / 'results' / 'merged'
ANALYSIS = ROOT / 'analysis'
ANALYSIS.mkdir(exist_ok=True)

DOMAIN_SIZES = {
    'medical': 5000,
    'legal': 3000,
    'code': 1000,
    'logic': 1500,
    'long_context': 902,
    # In-progress (sizes TBD; extract_curves uses len(results) when not in this dict)
    # 'math': N,
    # 'reasoning': N,
}
MODELS = ['gpt_4o', 'claude_sonnet_4_6', 'gemini_2_5_pro']

def find_file(domain: str, model: str, agents: int) -> Path | None:
    for suffix in (f'{agents}agent.json', f'{agents}agent_parallel.json',
                   f'{agents}agent_subprocess.json'):
        p = MERGED / f'{domain}_{model}_{suffix}'
        if p.exists():
            return p
    return None

def main():
    summary_rows = []
    per_q_rows = []
    for dom, expected in DOMAIN_SIZES.items():
        for model in MODELS:
            for agents in (1, 2, 3, 4):
                p = find_file(dom, model, agents)
                if p is None:
                    summary_rows.append({
                        'domain': dom, 'model': model, 'agent_count': agents,
                        'rows': 0, 'unique': 0, 'expected': expected,
                        'correct': None, 'accuracy': None, 'status': 'MISSING',
                        'file': None,
                    })
                    continue
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
                results = d.get('results', [])
                # dedup by question_id keeping last
                last_by_qid: dict[str, dict] = {}
                for r in results:
                    qid = r.get('question_id')
                    if qid is None:
                        continue
                    last_by_qid[qid] = r
                unique_results = list(last_by_qid.values())
                correct = sum(1 for r in unique_results if r.get('is_correct'))
                rows = len(results)
                unique = len(unique_results)
                acc = correct / unique * 100 if unique else 0.0

                if unique == expected:
                    status = 'OK'
                elif unique < expected:
                    status = f'SHORT_{unique}'
                elif unique > expected:
                    status = f'OVER_{unique}'  # shouldn't happen after dedup
                else:
                    status = 'OK'

                summary_rows.append({
                    'domain': dom, 'model': model, 'agent_count': agents,
                    'rows': rows, 'unique': unique, 'expected': expected,
                    'correct': correct, 'accuracy': acc, 'status': status,
                    'file': p.name,
                })

                for r in unique_results:
                    per_q_rows.append({
                        'domain': dom,
                        'model': model,
                        'agent_count': agents,
                        'question_id': r.get('question_id'),
                        'is_correct': bool(r.get('is_correct', False)),
                    })

    summary_df = pd.DataFrame(summary_rows)
    per_q_df = pd.DataFrame(per_q_rows)

    summary_df.to_csv(ANALYSIS / 'accuracy_table.csv', index=False, encoding='utf-8')
    per_q_df.to_csv(ANALYSIS / 'per_question.csv', index=False, encoding='utf-8')
    try:
        per_q_df.to_parquet(ANALYSIS / 'per_question.parquet', index=False)
    except Exception:
        pass

    # readable summary
    print('=== Summary table ===')
    pivot = summary_df.pivot_table(
        index=['domain', 'model'], columns='agent_count', values='accuracy',
    )
    print(pivot.round(2).to_string())
    print()

    print('=== Status counts ===')
    print(summary_df['status'].value_counts().to_string())
    print()

    issues = summary_df[summary_df['status'] != 'OK']
    if len(issues):
        print('=== Cells with issues ===')
        for _, r in issues.iterrows():
            print(f"  {r['domain']:14s} {r['model']:18s} {r['agent_count']}-agent  "
                  f"unique={r['unique']}/{r['expected']}  status={r['status']}  file={r['file']}")

    print()
    print(f"Wrote {ANALYSIS / 'accuracy_table.csv'}")
    print(f"Wrote {ANALYSIS / 'per_question.csv'}  ({len(per_q_df)} rows)")

if __name__ == '__main__':
    main()
