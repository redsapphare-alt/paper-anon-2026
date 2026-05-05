"""Compute decomposition (alpha, p, r, Delta) for tier-2 v2_FIXED data.

Using per-agent traces, we can pair questions across N values and compute:
  alpha = acc(N=1)
  p = P(correct@N | correct@1)
  r = P(correct@N | wrong@1)
  Delta = acc(N) - alpha = (1-alpha)*r - alpha*(1-p)

Verifies the decomposition identity holds (residual ~0).
Tests K-rule and alpha-rule prediction on the new domains.
"""
from __future__ import annotations
import json, sys, io, csv
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor_v2 import extract_math, extract_letter, normalise_number

T2 = Path(r'<paper_package>\results\v2\claudegeminimethreasoning')
OUT = Path(r'<paper_package>\analysis')


def reextract(r, domain):
    """Re-derive is_correct from full agent responses using v2 extractor."""
    ar = r.get('agent_responses', [])
    full_text = ' '.join((a.get('full_response') or '') for a in ar)
    if not full_text:
        full_text = (r.get('reasoning') or '') + ' ' + (r.get('predicted_answer') or '')
    gt = r.get('ground_truth') if 'ground_truth' in r else r.get('correct_answer')
    if domain == 'math':
        pred = extract_math(full_text)
        g = normalise_number(gt)
    else:
        pred = extract_letter(full_text, k=4)
        g = str(gt).strip().upper()[:1] if gt else ''
    return pred, g, (bool(pred) and pred == g)


# Load all 16 files into a {(domain, model, agents): {qid: is_correct}} dict
print('Loading and re-extracting tier-2 v2_FIXED data...')
data = {}
for sub in T2.iterdir():
    if not sub.is_dir() or not sub.name.startswith('v2_FIXED'): continue
    for f in sorted(sub.iterdir()):
        d = json.load(open(f, encoding='utf-8'))
        domain = d.get('domain')
        model = d.get('model')
        n_a = d.get('agent_count')
        results = d.get('results', [])
        per_q = {}
        for r in results:
            qid = r.get('question_id')
            _, _, ic = reextract(r, domain)
            per_q[qid] = ic
        data[(domain, model, n_a)] = per_q
        print(f'  {f.name}: n={len(per_q)}, acc={sum(per_q.values())/len(per_q):.3f}')

# Compute alpha, p, r for each (domain, model, N) cell with N>=2
print()
print('=== Decomposition (alpha, p, r, Delta) on tier-2 cells ===')
print(f'{"cell":35s} {"alpha":>6s} {"p":>6s} {"r":>6s} {"acc(N)":>7s} {"Delta_obs":>10s} {"Delta_pred":>10s} {"resid":>10s}')
print('-' * 100)
rows = []
for (dom, mod, n_a), per_q in data.items():
    if n_a == 1: continue
    one = data.get((dom, mod, 1))
    if not one: continue
    # Pair on shared qids
    shared = set(per_q.keys()) & set(one.keys())
    if not shared: continue
    # Counts
    n_shared = len(shared)
    alpha = sum(one[q] for q in shared) / n_shared
    accN = sum(per_q[q] for q in shared) / n_shared
    n_correct1 = sum(1 for q in shared if one[q])
    n_wrong1 = n_shared - n_correct1
    p = (sum(1 for q in shared if one[q] and per_q[q]) / n_correct1) if n_correct1 > 0 else 0
    r = (sum(1 for q in shared if (not one[q]) and per_q[q]) / n_wrong1) if n_wrong1 > 0 else 0
    delta_obs = accN - alpha
    delta_pred = (1 - alpha) * r - alpha * (1 - p)
    resid = delta_obs - delta_pred
    print(f'{dom:8s}/{mod:18s} N={n_a}  {alpha:6.3f} {p:6.3f} {r:6.3f} {accN:7.3f} '
          f'{delta_obs:+10.4f} {delta_pred:+10.4f} {resid:+10.4f}')
    rows.append({'domain': dom, 'model': mod, 'N': n_a, 'n_paired': n_shared,
                 'alpha': alpha, 'p': p, 'r': r, 'acc_N': accN,
                 'delta_obs': delta_obs, 'delta_pred': delta_pred, 'resid': resid})

# Save
out_csv = OUT / 'tier2_v2_decomposition.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
print(f'\\nWrote {out_csv}')

# K-rule and alpha-rule prediction test
print()
print('=== K-rule and alpha-rule on tier-2 (using rubric-estimated K) ===')
# Rough K estimates — gsm8k = arithmetic word problems (low K), arc-challenge = elementary science (low-mid K)
K_EST = {'math': 0.30, 'reasoning': 0.50}  # math < logic < reasoning ≈ logic
ALPHA_THRESH = 0.30  # from main 4-cell LOO

# Determine peak N* per cell
all_cells = defaultdict(dict)
for (dom, mod, n_a), per_q in data.items():
    all_cells[(dom, mod)][n_a] = sum(per_q.values()) / len(per_q) if per_q else float('nan')

print(f'{"cell":35s} {"alpha":>6s} {"K_est":>6s} {"peak":>5s} {"K-rule":>7s} {"a-rule":>7s}')
for (dom, mod), accs in sorted(all_cells.items()):
    if not accs: continue
    sorted_accs = sorted(accs.items())
    alpha = accs.get(1)
    peak_n = max(accs, key=accs.get)
    k_est = K_EST.get(dom, 0.5)
    k_rule = 1 if k_est >= 0.6 else 2
    a_rule = 1 if alpha is not None and alpha >= ALPHA_THRESH else 2
    target = 1 if peak_n == 1 else 2  # binarise peak to {1,2}
    k_ok = 'HIT' if k_rule == target else 'MISS'
    a_ok = 'HIT' if a_rule == target else 'MISS'
    print(f'{dom:8s}/{mod:18s}     {alpha:6.3f} {k_est:6.2f} N{peak_n:>3d}  N{k_rule}({k_ok})  N{a_rule}({a_ok})')
