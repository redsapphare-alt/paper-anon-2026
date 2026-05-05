"""GPT/Claude 실험 완전성 검증 - 각 (도메인, 모델, agent수)에 대해
실제 처리된 문제 수가 예상과 일치하는지, 정답 수와 정확도를 확인.
"""
import json
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DOMAIN_SIZES = {
    'medical': 5000,
    'legal': 3000,
    'code': 1000,
    'logic': 1500,
    'long_context': 902,
}
MODELS = {
    'gpt_4o': 'GPT-4o',
    'claude_sonnet_4_6': 'Claude 4.6',
}

merged = Path(__file__).resolve().parent.parent / 'results' / 'merged'

def find_file(domain, model, agents):
    for suffix in (f'{agents}agent.json', f'{agents}agent_parallel.json'):
        p = merged / f'{domain}_{model}_{suffix}'
        if p.exists():
            return p
    return None

def stats(p):
    d = json.load(open(p, encoding='utf-8'))
    res = d.get('results', [])
    return {
        'total_questions': d.get('total_questions', len(res)),
        'rows': len(res),
        'unique_qids': len({r.get('question_id') for r in res}),
        'correct': d.get('correct', sum(1 for r in res if r.get('is_correct'))),
        'accuracy': d.get('accuracy', 0),
        'status_field': d.get('status', '-'),
        'file': p.name,
    }

print(f"{'domain':<14}{'model':<12}{'agt':<5}{'rows':<7}{'uniq':<7}{'expt':<6}{'corr':<6}{'acc%':<7}{'note'}")
print('-' * 95)

issues = []
for dom, expected in DOMAIN_SIZES.items():
    for mkey, mname in MODELS.items():
        for a in (1, 2, 3, 4):
            p = find_file(dom, mkey, a)
            if not p:
                print(f"{dom:<14}{mname:<12}{a:<5}MISSING")
                issues.append(f"{dom}/{mname}/{a}-agent: file missing")
                continue
            s = stats(p)
            note = ''
            if s['rows'] == expected and s['unique_qids'] == expected:
                note = 'OK'
            elif s['rows'] > expected:
                note = f"DUPLICATES ({s['rows']} rows, {s['unique_qids']} unique)"
                issues.append(
                    f"{dom}/{mname}/{a}-agent: duplicates "
                    f"(rows={s['rows']}, unique={s['unique_qids']}, expected={expected})"
                )
            elif s['rows'] < expected:
                pct = s['rows'] / expected * 100
                note = f"SHORT {pct:.1f}%"
                issues.append(
                    f"{dom}/{mname}/{a}-agent: incomplete "
                    f"(rows={s['rows']}/{expected})"
                )
            else:
                note = 'OK?'
            print(
                f"{dom:<14}{mname:<12}{a:<5}{s['rows']:<7}{s['unique_qids']:<7}"
                f"{expected:<6}{s['correct']:<6}{s['accuracy']:<7.2f}{note}"
            )

print()
print(f'=== {len(issues)} issues found ===')
for x in issues:
    print(' -', x)
