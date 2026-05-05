"""실험 결과 JSON의 `reasoning` 필드 길이로 cognitive load proxy 계산.

reasoning 필드는 모델이 답을 도출하기 위해 생성한 텍스트(chain-of-thought) — 길이가
길수록 답에 도달하기 위한 추론 단계가 많거나 복잡함을 의미. 이는 task의 인지적 깊이
지표 (D, K와 부분 상관 가능).

is_correct 판정 버그와 무관 — reasoning 텍스트 자체는 모델 출력이므로 신뢰 가능.

출력:
  analysis/reasoning_depth.json
  - 도메인별·모델별·agent수별 reasoning 길이 분포
  - 1-agent 기준 평균을 도메인의 baseline cognitive depth로 사용 (정의)
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'results' / 'merged'

DOMAIN_SIZES = {'medical': 5000, 'legal': 3000, 'code': 1000, 'logic': 1500, 'long_context': 902}

def parse_name(name):
    base = name.replace('.json', '')
    parts = base.split('_')
    a_idx = next((i for i, p in enumerate(parts) if p.endswith('agent') and p[:-5].isdigit()), None)
    if a_idx is None: return None
    agents = int(parts[a_idx][:-5])
    for dom in sorted(DOMAIN_SIZES, key=lambda x: -x.count('_')):
        prefix = dom.split('_')
        if parts[:len(prefix)] == prefix:
            return dom, '_'.join(parts[len(prefix):a_idx]), agents
    return None

def quantiles(values):
    if not values:
        return None
    s = sorted(values)
    return {
        'mean': st.mean(s), 'median': st.median(s),
        'p25': s[int(0.25 * len(s))], 'p75': s[int(0.75 * len(s))],
        'p95': s[int(0.95 * len(s))], 'max': max(s), 'n': len(s),
    }

def main():
    by_cell = defaultdict(list)
    for f in sorted(M.glob('*.json')):
        parsed = parse_name(f.name)
        if not parsed: continue
        dom, model, agents = parsed
        try:
            d = json.load(open(f, encoding='utf-8'))
        except: continue
        for r in d.get('results', []):
            rs = r.get('reasoning')
            if not rs: continue
            by_cell[(dom, model, agents)].append(len(rs))

    cell_stats = {}
    for (dom, model, agents), lens in by_cell.items():
        cell_stats[f'{dom}/{model}/{agents}agent'] = quantiles(lens)

    # Domain-level baseline: 1-agent reasoning length averaged over models
    domain_baseline = {}
    for dom in DOMAIN_SIZES:
        agent1_means = []
        all_means = []
        for (d, m, a), lens in by_cell.items():
            if d != dom or not lens: continue
            if a == 1:
                agent1_means.append(st.mean(lens))
            all_means.append(st.mean(lens))
        if agent1_means:
            domain_baseline[dom] = {
                '1agent_avg_reasoning_chars': st.mean(agent1_means),
                '1agent_avg_reasoning_tokens_est': st.mean(agent1_means) // 4,
                'all_agents_avg_reasoning_chars': st.mean(all_means) if all_means else None,
                'n_models_with_1agent': len(agent1_means),
            }

    # Normalize cognitive depth to [0,1]: rank-normalize by domain mean
    if len(domain_baseline) >= 2:
        means = [v['1agent_avg_reasoning_chars'] for v in domain_baseline.values()]
        lo, hi = min(means), max(means)
        rng = hi - lo if hi > lo else 1
        for dom, v in domain_baseline.items():
            v['cognitive_depth_normalized'] = (v['1agent_avg_reasoning_chars'] - lo) / rng

    out = {
        '_meta': {
            'description': 'Average length of reasoning text the model generated (per single agent).'
                            ' Proxy for cognitive depth required by the task.',
            'caveat': 'reasoning field present mainly in GPT-4o files; absent in many Claude files.'
                      ' Domain comparison should use only domains where multiple models agree.',
            'normalization': 'cognitive_depth_normalized = (mean - min) / (max - min) across domains'
                             ' using 1-agent averages.',
        },
        'domain_baseline': domain_baseline,
        'per_cell': cell_stats,
    }

    OUT = ROOT / 'analysis' / 'reasoning_depth.json'
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=== Domain-level reasoning depth (1-agent average) ===')
    for dom, v in sorted(domain_baseline.items(), key=lambda x: -x[1]['1agent_avg_reasoning_chars']):
        print(f"  {dom:14s}  avg_chars={v['1agent_avg_reasoning_chars']:8.0f}  "
              f"~tokens={v['1agent_avg_reasoning_tokens_est']:5.0f}  "
              f"n_models={v['n_models_with_1agent']}  "
              f"depth_norm={v.get('cognitive_depth_normalized', 0):.2f}")

    print(f"\nWrote {OUT}")
    print(f"  per-cell entries: {len(cell_stats)}")

if __name__ == '__main__':
    main()
