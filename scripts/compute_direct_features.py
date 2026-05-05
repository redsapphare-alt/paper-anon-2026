"""데이터에서 직접 측정 가능한 task feature 계산.

LLM judge 없이 즉시 가능한 feature:
- question 평균 길이 (chars / 추정 tokens)
- context 평균 길이 (long_context의 경우 별도 context 필드)
- answer space (binary / MCQ-N / free-form)
- ground_truth 길이 분포
- (실험 결과에서 추출) 평균 reasoning 길이 — agent당 추론 깊이의 근사치

출력:
  analysis/direct_features.json  — 도메인별 직접 측정 feature
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
from collections import Counter

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent

# 도메인별 데이터 파일 (사용된 실험 기준)
DATA_FILES = {
    'medical': ROOT / 'data/medical/medqa_5000.json',
    'legal':   ROOT / 'data/legal/legalbench_3000.json',
    'code':    ROOT / 'data/code/code_combined_1000.json',  # 실험은 1000 사용 추정
    'logic':   ROOT / 'data/logic/mmlu_logic_1500.json',
    'long_context': ROOT / 'data/long_context/sampled/long_context_small_1000.json',
}

# fallback: 1000짜리 사용
DATA_FILES_FALLBACK = {
    'medical': ROOT / 'data/medical/medqa_1000.json',
    'legal':   ROOT / 'data/legal/legalbench_1000.json',
    'code':    ROOT / 'data/code/code_combined_1000.json',
    'logic':   ROOT / 'data/logic/mmlu_logic_500.json',
    'long_context': ROOT / 'data/long_context/sampled/long_context_small_1000.json',
}

def load_first(domain):
    for cand in (DATA_FILES.get(domain), DATA_FILES_FALLBACK.get(domain)):
        if cand and cand.exists():
            return cand, json.load(open(cand, encoding='utf-8'))
    raise FileNotFoundError(domain)

def estimate_tokens(s: str) -> int:
    """Heuristic: ~4 chars per token for English."""
    return max(1, len(s) // 4)

def get_question_text(item: dict, domain: str) -> str:
    return item.get('question', '')

def get_context_text(item: dict, domain: str) -> str:
    return item.get('context', '') or ''

def get_options(item: dict, domain: str):
    """Returns list of options if MCQ, else None."""
    if 'options' in item and isinstance(item['options'], dict):
        return list(item['options'].values())
    if 'choices' in item and isinstance(item['choices'], list):
        return item['choices']
    if 'choice_texts' in item and isinstance(item['choice_texts'], list):
        return item['choice_texts']
    return None

def get_answer(item: dict, domain: str):
    return item.get('answer')

def main():
    out = {}
    for dom in ['medical', 'legal', 'code', 'logic', 'long_context']:
        path, data = load_first(dom)
        n = len(data)
        # sample 500 to keep compute light
        sample_size = min(500, n)
        sample = data[:sample_size]

        q_lens = [len(get_question_text(it, dom)) for it in sample]
        c_lens = [len(get_context_text(it, dom)) for it in sample]
        opts_count_dist = Counter(len(get_options(it, dom)) if get_options(it, dom) else 0 for it in sample)

        ans_samples = [get_answer(it, dom) for it in sample[:50]]
        ans_lens = [len(str(a)) for a in ans_samples if a is not None]

        # answer space classification
        opts_present = sum(1 for it in sample if get_options(it, dom))
        if opts_present > sample_size * 0.5:
            common_n = opts_count_dist.most_common(1)[0][0]
            answer_space = f'mcq_{common_n}'
        else:
            # check if answer is short categorical (binary etc.)
            unique_ans = set(str(get_answer(it, dom)).strip() for it in data[:200] if get_answer(it, dom))
            if len(unique_ans) <= 6 and all(len(str(a)) <= 20 for a in unique_ans):
                answer_space = f'categorical_{len(unique_ans)}'
            else:
                answer_space = 'free_form'

        out[dom] = {
            'data_file': str(path.relative_to(ROOT)),
            'n_total': n,
            'n_sampled_for_features': sample_size,
            'question_chars': {
                'median': st.median(q_lens),
                'mean': st.mean(q_lens),
                'p95': sorted(q_lens)[int(0.95 * len(q_lens))],
                'max': max(q_lens),
            },
            'question_tokens_est': {
                'median': estimate_tokens('x' * int(st.median(q_lens))),
                'mean': estimate_tokens('x' * int(st.mean(q_lens))),
            },
            'context_chars': {
                'median': st.median(c_lens),
                'mean': st.mean(c_lens),
                'max': max(c_lens),
                'present_pct': sum(1 for c in c_lens if c > 0) / len(c_lens) * 100,
            },
            'answer_space': answer_space,
            'options_distribution': dict(opts_count_dist),
            'ground_truth_chars': {
                'median': st.median(ans_lens) if ans_lens else 0,
                'mean': st.mean(ans_lens) if ans_lens else 0,
                'max': max(ans_lens) if ans_lens else 0,
            } if ans_lens else None,
        }

    OUT = ROOT / 'analysis' / 'direct_features.json'
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=== Direct Features ===\n')
    for dom, feats in out.items():
        print(f'## {dom}')
        print(f'  data: {feats["data_file"]}  (n={feats["n_total"]})')
        q = feats['question_chars']
        print(f'  question chars: median={q["median"]:.0f}  mean={q["mean"]:.0f}  '
              f'p95={q["p95"]}  max={q["max"]}')
        print(f'  question tokens (est): median={feats["question_tokens_est"]["median"]}  '
              f'mean={feats["question_tokens_est"]["mean"]}')
        c = feats['context_chars']
        print(f'  context chars:  median={c["median"]:.0f}  mean={c["mean"]:.0f}  '
              f'max={c["max"]}  present={c["present_pct"]:.0f}%')
        print(f'  answer space: {feats["answer_space"]}')
        if feats['ground_truth_chars']:
            g = feats['ground_truth_chars']
            print(f'  GT chars: median={g["median"]:.0f}  mean={g["mean"]:.0f}  max={g["max"]}')
        print()

    print(f'Wrote {OUT}')

if __name__ == '__main__':
    main()
