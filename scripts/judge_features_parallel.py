"""병렬 question-level LLM judging — 호출당 시간이 길 때 동시 호출로 가속.

기존 judge_features.py의 question-level이 sequential이라 도메인당 13분 걸리는 문제 해결.
ThreadPoolExecutor로 N_WORKERS개 동시 호출, 도메인당 ~75초로 단축.

사용:
  python scripts/judge_features_parallel.py --judges gemini --workers 10 --n-questions 50
  python scripts/judge_features_parallel.py --judges gpt --workers 8 --n-questions 50
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse from judge_features.py (it sets up UTF-8 stdout itself)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_features import (
    load_env_file, JUDGES, DOMAINS, FEATURE_PROMPT_PREFIX,
    QUESTION_LEVEL_PROMPT_TEMPLATE, render_item, parse_scores,
)

ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = ROOT / 'analysis' / 'feature_judge'
JUDGE_DIR.mkdir(parents=True, exist_ok=True)

load_env_file()


def judge_one(idx, item, dom, dom_desc, judge_name, model):
    prompt = QUESTION_LEVEL_PROMPT_TEMPLATE.format(
        prefix=FEATURE_PROMPT_PREFIX, dom_name=dom, dom_desc=dom_desc,
        item=render_item(item, dom),
    )
    try:
        text = JUDGES[judge_name]['fn'](prompt, model=model)
        parsed = parse_scores(text)
        return {'idx': idx, 'item_id': item.get('id'), 'parsed': parsed, 'response': text}
    except Exception as e:
        return {'idx': idx, 'item_id': item.get('id'), 'parsed': None, 'error': str(e)[:300]}


def question_level_parallel(dom, judge_name, n_questions, workers, model, seed=42):
    info = DOMAINS[dom]
    data = json.load(open(info['data_file'], encoding='utf-8'))
    rng = random.Random(seed)
    sample_items = rng.sample(data, k=min(n_questions, len(data)))

    results = [None] * len(sample_items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(judge_one, i, it, dom, info['description'], judge_name, model): i
            for i, it in enumerate(sample_items)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(sample_items):
                ok = sum(1 for r in results if r and r.get('parsed'))
                print(f"  {judge_name}/{dom}: {done}/{len(sample_items)} done ({ok} parsed OK)")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--judges', default='gemini', choices=['claude', 'gpt', 'gemini', 'both', 'all'])
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--n-questions', type=int, default=50)
    p.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default=None, help='override default judge model')
    p.add_argument('--skip-existing', action='store_true', help='skip if output file exists')
    args = p.parse_args()

    if args.judges == 'both':
        judges_to_run = ['gpt', 'gemini']
    elif args.judges == 'all':
        judges_to_run = ['claude', 'gpt', 'gemini']
    else:
        judges_to_run = [args.judges]

    available = []
    for j in judges_to_run:
        env = JUDGES[j]['env']
        if os.environ.get(env):
            available.append(j); continue
        if j == 'gemini' and (ROOT / '.env.txt').exists():
            available.append(j); continue
        print(f"⚠️  Skipping judge '{j}': {env} not set")
    if not available:
        sys.exit("No API keys available")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Workers: {args.workers}, Judges: {available}, Domains: {args.domains}, N: {args.n_questions}")
    print()

    for j in available:
        model = args.model or JUDGES[j]['default_model']
        for dom in args.domains:
            out_file = JUDGE_DIR / f'{j}_question-level_{dom}_{timestamp}.json'
            # check existing files for this judge×domain
            existing = sorted(JUDGE_DIR.glob(f'{j}_question-level_{dom}_*.json'))
            if args.skip_existing and existing:
                # confirm complete
                last = json.load(open(existing[-1], encoding='utf-8'))
                ok = sum(1 for r in last if r.get('parsed'))
                if ok >= args.n_questions:
                    print(f"[skip] {j}/{dom}: existing file has {ok} parsed OK")
                    continue
            import time
            t0 = time.time()
            print(f"\n=== {j} / {dom} (model={model}, workers={args.workers}) ===")
            results = question_level_parallel(dom, j, args.n_questions, args.workers, model, args.seed)
            elapsed = time.time() - t0
            ok = sum(1 for r in results if r.get('parsed'))
            err = sum(1 for r in results if r.get('error'))
            print(f"  done: {ok}/{len(results)} parsed, {err} errors, {elapsed:.1f}s")
            out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\nWrote outputs to {JUDGE_DIR}/<judge>_question-level_<domain>_{timestamp}.json")


if __name__ == '__main__':
    main()
