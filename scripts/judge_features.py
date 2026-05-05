"""LLM-as-judge로 각 도메인의 (D, V, K, S) feature 점수화.

두 가지 모드:
  --domain-level: 도메인 정의 + 샘플 5문제를 judge에게 보여주고 도메인 점수 매김 (3 회 반복)
  --question-level: 도메인당 N문제 sample, 각 question에 점수 매김

사용:
  ANTHROPIC_API_KEY=sk-... python scripts/judge_features.py --judges claude --mode domain-level
  OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-... python scripts/judge_features.py --judges both --mode domain-level
  python scripts/judge_features.py --judges both --mode question-level --n-questions 50

API 키가 없으면 dry-run 모드로 prompt만 출력.

의존성: anthropic, openai (자동 import; 없으면 그 judge 건너뜀)
출력:
  analysis/feature_judge/<judge>_<mode>_<timestamp>.json — 모든 judge 응답
  analysis/domain_features.json — 최종 평균/통합 결과
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / 'analysis'
JUDGE_DIR = ANALYSIS / 'feature_judge'
JUDGE_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file():
    """Load KEY=VALUE pairs from .env.txt into os.environ if not already set.

    Backwards compatible: if file is a single bare token (legacy format),
    treat it as GEMINI_API_KEY.
    """
    env_file = ROOT / '.env.txt'
    if not env_file.exists():
        return
    text = env_file.read_text(encoding='utf-8').strip()
    if not text:
        return
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')]
    # Legacy: single bare token
    if len(lines) == 1 and '=' not in lines[0]:
        os.environ.setdefault('GEMINI_API_KEY', lines[0])
        return
    # alias map: same key acceptable under multiple names
    aliases = {
        'CLAUDE_API_KEY': 'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY': 'GEMINI_API_KEY',
    }
    for line in lines:
        if '=' not in line:
            continue
        k, _, v = line.partition('=')
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or not v:
            continue
        if not os.environ.get(k):
            os.environ[k] = v
        # Also set the canonical name if this is an alias
        canonical = aliases.get(k)
        if canonical and not os.environ.get(canonical):
            os.environ[canonical] = v
        # Reverse: if user used canonical name, propagate to aliases too (low priority)
        for alias_name, alias_target in aliases.items():
            if k == alias_target and not os.environ.get(alias_name):
                os.environ[alias_name] = v


load_env_file()

# 도메인 정의 + 데이터셋 location
DOMAINS = {
    'medical': {
        'description': "MedQA-style USMLE clinical vignettes. Each item is a long patient case "
                       "(symptoms, history, exam) followed by a 4-option multiple-choice question "
                       "asking the most appropriate diagnosis or next step in management. Single correct answer letter.",
        'data_file': ROOT / 'data/medical/medqa_5000.json',
    },
    'legal': {
        'description': "LegalBench tasks, primarily privacy_policy_qa and contract_nli. Each item is a short text "
                       "snippet from a legal document, and the task is binary classification (Relevant/Irrelevant or Yes/No). "
                       "Requires legal-domain interpretation.",
        'data_file': ROOT / 'data/legal/legalbench_3000.json',
    },
    'code': {
        'description': "HumanEval-style code generation. Each item gives a Python function signature plus docstring "
                       "(possibly with examples) and the task is to write the function body. Verified by hidden unit tests.",
        'data_file': ROOT / 'data/code/code_combined_1000.json',
    },
    'logic': {
        'description': "MMLU formal logic / philosophy / logical fallacies tasks. Each item is a short argument "
                       "or logical statement, with a 4-option multiple-choice question about identifying premises, "
                       "conclusions, fallacies, or valid inferences.",
        'data_file': ROOT / 'data/logic/mmlu_logic_1500.json',
    },
    'long_context': {
        'description': "Reading comprehension over long passages (QuALITY dataset). Each item has a long passage "
                       "(median ~26,000 characters) plus a 4-option multiple-choice question. Answer is in the passage.",
        'data_file': ROOT / 'data/long_context/sampled/long_context_small_1000.json',
    },
}

# Feature definitions (mirror feature_judge_protocol.md)
FEATURE_PROMPT_PREFIX = """You are evaluating cognitive task properties on four orthogonal axes.

For each axis, output a number in [0.0, 1.0] (one decimal place is fine):

1) Decomposability (D): Can the task be split into INDEPENDENT sub-tasks that can be solved
   in parallel and combined into the final answer?
   0.0 = unitary integrative reasoning required (e.g., medical diagnosis from many symptoms)
   1.0 = highly decomposable (e.g., independent module implementation, parallel passage reading)

2) Verifiability (V): How deterministically can the answer be CHECKED?
   0.0 = subjective judgment (e.g., creative writing quality)
   0.6-0.8 = categorical match (MCQ letter, classification label)
   1.0 = automatic execution-based verification (unit tests for code, math equality)

3) Knowledge concentration (K): How much does answering depend on NARROW specialized
   domain expertise (vs general reasoning or info given in the prompt)?
   0.0 = answer is in the provided text/general knowledge
   0.5 = moderate domain (general STEM, formal logic)
   1.0 = very narrow expert knowledge (specific case law, niche pharmacology)

4) Solution diversity (S): Are MULTIPLE valid solution paths available?
   0.0 = essentially one path (single fact recall, single classification)
   0.5 = some diversity (different reasoning routes to same answer)
   1.0 = many valid solutions (creative tasks, multiple algorithms)

Output format: ONLY JSON like {"D": 0.3, "V": 0.9, "K": 0.8, "S": 0.3}. No prose."""

DOMAIN_LEVEL_PROMPT_TEMPLATE = """{prefix}

---
TASK DOMAIN TO EVALUATE: {dom_name}

Description: {dom_desc}

Five sample items from this dataset (questions truncated for brevity):

{samples}

Now provide D, V, K, S for THIS DOMAIN OVERALL based on the description and samples."""

QUESTION_LEVEL_PROMPT_TEMPLATE = """{prefix}

---
DOMAIN: {dom_name} ({dom_desc})

Single item to evaluate:

{item}

Provide D, V, K, S for THIS SPECIFIC item."""


def render_item(item: dict, dom: str, max_chars: int = 800) -> str:
    """Render one dataset item as a compact text block."""
    parts = []
    q = item.get('question', '')
    if dom == 'long_context':
        ctx = item.get('context', '')
        if ctx:
            parts.append(f"[passage, {len(ctx)} chars] {ctx[:200]}…[truncated]")
        parts.append(f"[question] {q}")
    else:
        parts.append(f"[question] {q[:max_chars]}")
    if 'options' in item and isinstance(item['options'], dict):
        opts = "\n".join(f"  {k}. {v}" for k, v in item['options'].items())
        parts.append(f"[options]\n{opts}")
    elif 'choices' in item and isinstance(item['choices'], list):
        opts = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(item['choices']))
        parts.append(f"[options]\n{opts}")
    elif 'choice_texts' in item:
        opts = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(item['choice_texts']))
        parts.append(f"[options]\n{opts}")
    return "\n".join(parts)


def parse_scores(text: str) -> dict | None:
    """Extract {D,V,K,S} dict from model output."""
    m = re.search(r'\{[^}]*"D"[^}]+\}', text, re.DOTALL)
    if not m: return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ('D', 'V', 'K', 'S'):
        if k not in d: return None
        try:
            v = float(d[k])
            if not (0 <= v <= 1): return None
            out[k] = v
        except (TypeError, ValueError):
            return None
    return out


# ---- API callers ----
def call_anthropic(prompt: str, model: str = 'claude-sonnet-4-5-20250929') -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed: pip install anthropic")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    # Extract first text block (skip thinking blocks if any)
    for block in resp.content:
        if getattr(block, 'type', None) == 'text':
            return block.text
    return resp.content[0].text if resp.content else ''

def call_openai(prompt: str, model: str = 'gpt-4o') -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed: pip install openai")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, max_tokens=200, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

_GEMINI_CONFIGURED = False
def call_gemini(prompt: str, model: str = 'gemini-2.5-pro') -> str:
    global _GEMINI_CONFIGURED
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("google-generativeai not installed: pip install google-generativeai")
    if not _GEMINI_CONFIGURED:
        key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not key:
            env_file = ROOT / '.env.txt'
            if env_file.exists():
                key = env_file.read_text().strip()
        if not key:
            raise RuntimeError("No Gemini API key (set GOOGLE_API_KEY or write to .env.txt)")
        genai.configure(api_key=key)
        _GEMINI_CONFIGURED = True
    m = genai.GenerativeModel(model)
    # 2.5-pro reserves output tokens for "thinking" → must allocate generously
    resp = m.generate_content(
        prompt,
        generation_config={'temperature': 0.0, 'max_output_tokens': 4096},
    )
    # Try multiple ways to extract text safely
    try:
        if resp.text:
            return resp.text
    except (ValueError, AttributeError):
        pass
    # Fallback: walk candidates → parts
    out = []
    for cand in (resp.candidates or []):
        content = getattr(cand, 'content', None)
        if not content: continue
        for part in (getattr(content, 'parts', None) or []):
            t = getattr(part, 'text', None)
            if t: out.append(t)
    if out:
        return ''.join(out)
    # Diagnostic info
    finish = getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else '?'
    raise RuntimeError(f"Empty Gemini response (finish_reason={finish})")


JUDGES = {
    'claude': {
        'fn': call_anthropic,
        'env': 'ANTHROPIC_API_KEY',
        'default_model': 'claude-sonnet-4-5-20250929',
    },
    'gpt': {
        'fn': call_openai,
        'env': 'OPENAI_API_KEY',
        'default_model': 'gpt-4o',
    },
    'gemini': {
        'fn': call_gemini,
        'env': 'GEMINI_API_KEY',  # also accepts .env.txt fallback
        'default_model': 'gemini-2.5-flash',  # Pro is ~27s/call due to thinking; flash is much faster
    },
}


def domain_level_judge(dom: str, judge_name: str, repetitions: int = 3, model: str | None = None,
                        dry_run: bool = False, seed: int = 42):
    info = DOMAINS[dom]
    data = json.load(open(info['data_file'], encoding='utf-8'))
    rng = random.Random(seed)
    sample_items = rng.sample(data, k=min(5, len(data)))
    samples_text = "\n---\n".join(render_item(it, dom) for it in sample_items)

    prompt = DOMAIN_LEVEL_PROMPT_TEMPLATE.format(
        prefix=FEATURE_PROMPT_PREFIX,
        dom_name=dom,
        dom_desc=info['description'],
        samples=samples_text,
    )

    if dry_run:
        print(f"\n[DRY RUN] judge={judge_name} domain={dom}")
        print("Prompt length:", len(prompt))
        print("First 1000 chars:", prompt[:1000])
        return [{'rep': i, 'response': None, 'parsed': None, 'dry_run': True} for i in range(repetitions)]

    judge = JUDGES[judge_name]
    use_model = model or judge['default_model']
    results = []
    for rep in range(repetitions):
        try:
            text = judge['fn'](prompt, model=use_model)
            parsed = parse_scores(text)
            results.append({'rep': rep, 'response': text, 'parsed': parsed})
            print(f"  [{judge_name}/{dom}/rep{rep}] parsed={parsed}")
        except Exception as e:
            print(f"  [{judge_name}/{dom}/rep{rep}] ERROR: {e}")
            results.append({'rep': rep, 'response': None, 'parsed': None, 'error': str(e)})
        time.sleep(0.5)
    return results


def question_level_judge(dom: str, judge_name: str, n_questions: int = 50, model: str | None = None,
                          dry_run: bool = False, seed: int = 42):
    info = DOMAINS[dom]
    data = json.load(open(info['data_file'], encoding='utf-8'))
    rng = random.Random(seed)
    sample_items = rng.sample(data, k=min(n_questions, len(data)))

    if dry_run:
        print(f"\n[DRY RUN] judge={judge_name} domain={dom} mode=question-level n={len(sample_items)}")
        print("First prompt length:", len(QUESTION_LEVEL_PROMPT_TEMPLATE.format(
            prefix=FEATURE_PROMPT_PREFIX, dom_name=dom, dom_desc=info['description'],
            item=render_item(sample_items[0], dom))))
        return []

    judge = JUDGES[judge_name]
    use_model = model or judge['default_model']
    results = []
    for i, it in enumerate(sample_items):
        prompt = QUESTION_LEVEL_PROMPT_TEMPLATE.format(
            prefix=FEATURE_PROMPT_PREFIX, dom_name=dom, dom_desc=info['description'],
            item=render_item(it, dom),
        )
        try:
            text = judge['fn'](prompt, model=use_model)
            parsed = parse_scores(text)
            results.append({'idx': i, 'item_id': it.get('id'), 'parsed': parsed, 'response': text})
        except Exception as e:
            results.append({'idx': i, 'item_id': it.get('id'), 'parsed': None, 'error': str(e)})
        if (i + 1) % 10 == 0:
            print(f"  {judge_name}/{dom}: {i+1}/{len(sample_items)} done")
        time.sleep(0.3)
    return results


def aggregate(domain_results: dict) -> dict:
    """Aggregate per-judge per-rep parsed scores → per-domain mean/std per axis."""
    out = {}
    for dom, by_judge in domain_results.items():
        row = {'judges': {}}
        all_scores = {'D': [], 'V': [], 'K': [], 'S': []}
        for judge_name, reps in by_judge.items():
            ds, vs, ks, ss = [], [], [], []
            for r in reps:
                p = r.get('parsed')
                if not p: continue
                ds.append(p['D']); vs.append(p['V']); ks.append(p['K']); ss.append(p['S'])
            if ds:
                row['judges'][judge_name] = {
                    'D_mean': sum(ds)/len(ds), 'V_mean': sum(vs)/len(vs),
                    'K_mean': sum(ks)/len(ks), 'S_mean': sum(ss)/len(ss),
                    'n_reps_parsed': len(ds),
                }
                all_scores['D'] += ds; all_scores['V'] += vs
                all_scores['K'] += ks; all_scores['S'] += ss
        if all_scores['D']:
            row['combined'] = {
                'D': sum(all_scores['D'])/len(all_scores['D']),
                'V': sum(all_scores['V'])/len(all_scores['V']),
                'K': sum(all_scores['K'])/len(all_scores['K']),
                'S': sum(all_scores['S'])/len(all_scores['S']),
                'n_total_reps': len(all_scores['D']),
            }
        out[dom] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--judges', default='gemini', choices=['claude', 'gpt', 'gemini', 'both', 'all'])
    p.add_argument('--mode', default='domain-level', choices=['domain-level', 'question-level'])
    p.add_argument('--repetitions', type=int, default=3, help='reps per judge×domain (domain-level only)')
    p.add_argument('--n-questions', type=int, default=50, help='questions per domain (question-level only)')
    p.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.judges == 'both':
        judges_to_run = ['gpt', 'gemini']  # default cross-judge pair (Claude often gated)
    elif args.judges == 'all':
        judges_to_run = ['claude', 'gpt', 'gemini']
    else:
        judges_to_run = [args.judges]

    # Filter by API key availability (gemini also accepts .env.txt fallback)
    available = []
    for j in judges_to_run:
        if args.dry_run:
            available.append(j); continue
        env = JUDGES[j]['env']
        if os.environ.get(env):
            available.append(j); continue
        if j == 'gemini':
            env_file = ROOT / '.env.txt'
            if env_file.exists():
                available.append(j); continue
        print(f"⚠️  Skipping judge '{j}': {env} not set")

    if not available and not args.dry_run:
        print("\nNo API keys available. Either:")
        print(f"  set ANTHROPIC_API_KEY for claude judge, OPENAI_API_KEY for gpt judge")
        print(f"  or run with --dry-run to inspect prompts")
        sys.exit(1)

    print(f"Mode: {args.mode}")
    print(f"Judges: {available}")
    print(f"Domains: {args.domains}")
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.mode == 'domain-level':
        all_results = {dom: {} for dom in args.domains}
        for j in available:
            for dom in args.domains:
                print(f"\n=== {j} / {dom} ===")
                results = domain_level_judge(dom, j, repetitions=args.repetitions,
                                              dry_run=args.dry_run, seed=args.seed)
                all_results[dom][j] = results
                # save raw per judge
                out = JUDGE_DIR / f'{j}_domain-level_{dom}_{timestamp}.json'
                out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

        agg = aggregate(all_results)
        out_main = ANALYSIS / 'domain_features.json'
        out_main.write_text(json.dumps({
            '_meta': {'mode': args.mode, 'judges': available, 'reps': args.repetitions,
                       'timestamp': timestamp, 'dry_run': args.dry_run},
            'aggregated': agg,
            'raw_per_domain_judge': all_results,
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nWrote {out_main}")
        if not args.dry_run:
            print("\n=== Aggregated scores ===")
            for dom, row in agg.items():
                if 'combined' in row:
                    c = row['combined']
                    print(f"  {dom:14s}  D={c['D']:.2f}  V={c['V']:.2f}  K={c['K']:.2f}  S={c['S']:.2f}  (n={c['n_total_reps']})")

    else:
        all_q = {dom: {} for dom in args.domains}
        for j in available:
            for dom in args.domains:
                print(f"\n=== {j} / {dom} (question-level n={args.n_questions}) ===")
                results = question_level_judge(dom, j, n_questions=args.n_questions,
                                                dry_run=args.dry_run, seed=args.seed)
                all_q[dom][j] = results
                out = JUDGE_DIR / f'{j}_question-level_{dom}_{timestamp}.json'
                out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

        # Per-domain summary
        out_main = ANALYSIS / 'domain_features_question_level.json'
        summary = {}
        for dom, by_j in all_q.items():
            summary[dom] = {}
            for j, items in by_j.items():
                ds = [r['parsed']['D'] for r in items if r.get('parsed')]
                vs = [r['parsed']['V'] for r in items if r.get('parsed')]
                ks = [r['parsed']['K'] for r in items if r.get('parsed')]
                ss = [r['parsed']['S'] for r in items if r.get('parsed')]
                if ds:
                    summary[dom][j] = {
                        'D_mean': sum(ds)/len(ds), 'D_std': (sum((d - sum(ds)/len(ds))**2 for d in ds) / len(ds)) ** 0.5,
                        'V_mean': sum(vs)/len(vs),
                        'K_mean': sum(ks)/len(ks),
                        'S_mean': sum(ss)/len(ss),
                        'n_questions_parsed': len(ds),
                    }
        out_main.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"\nWrote {out_main}")


if __name__ == '__main__':
    main()
