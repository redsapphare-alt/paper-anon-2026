"""각 도메인에서 N개 질문 random sample 추출 (LLM judge question-level용).

LLM judge가 question-level 점수를 매길 때 같은 sample을 사용하도록 결정적 시드 사용.
샘플의 features (text, options, etc.)도 정리해서 judge에 보내기 좋은 포맷으로 저장.

출력:
  analysis/sampled_questions/{domain}_n{N}.json
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'sampled_questions'
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = {
    'medical': ROOT / 'data/medical/medqa_5000.json',
    'legal':   ROOT / 'data/legal/legalbench_3000.json',
    'code':    ROOT / 'data/code/code_combined_1000.json',
    'logic':   ROOT / 'data/logic/mmlu_logic_1500.json',
    'long_context': ROOT / 'data/long_context/sampled/long_context_small_1000.json',
}

N_SAMPLE = 200
SEED = 42

def main():
    for dom, path in DOMAINS.items():
        data = json.load(open(path, encoding='utf-8'))
        rng = random.Random(SEED)
        sample = rng.sample(data, k=min(N_SAMPLE, len(data)))

        # Compact form for judge
        compact = []
        for it in sample:
            entry = {
                'id': it.get('id', it.get('task_id', '?')),
                'question': it.get('question', '')[:1500],  # cap at 1500 chars
            }
            if dom == 'long_context':
                ctx = it.get('context', '')
                # truncate context for judge prompt; judge can still tell it's long
                entry['context_length_chars'] = len(ctx)
                entry['context_preview'] = ctx[:500] + ('…[truncated]' if len(ctx) > 500 else '')
            if 'options' in it and isinstance(it['options'], dict):
                entry['options'] = it['options']
            elif 'choices' in it and isinstance(it['choices'], list):
                entry['choices'] = it['choices']
            elif 'choice_texts' in it:
                entry['choices'] = it['choice_texts']
            entry['answer'] = it.get('answer')
            compact.append(entry)

        out_file = OUT / f'{dom}_n{N_SAMPLE}.json'
        out_file.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  {dom:14s}: {len(compact)} samples -> {out_file.relative_to(ROOT)}")

if __name__ == '__main__':
    main()
