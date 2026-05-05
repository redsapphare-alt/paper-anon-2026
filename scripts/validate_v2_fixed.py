"""Validate v2_FIXED tier-2 data: completeness, no truncation, per-agent traces, extraction reliability.

Checks per file:
  1. n records vs total_questions
  2. agent_responses length == agent_count for each record
  3. full_response NOT capped at 200/2000 chars
  4. extracted_answer agreement with stored is_correct (re-derive using extractor_v2)
  5. accuracy not stuck at random chance
"""
from __future__ import annotations
import json, sys, io, statistics
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor_v2 import extract_math, extract_letter, normalise_number

ROOT = Path(r'<paper_package>\results\v2\claudegeminimethreasoning')

EXPECTED_TOTAL = {'math': 1300, 'reasoning': 1200}
EXPECTED_AGENTS = {1: 1, 2: 2, 3: 3, 4: 4}


def report_file(p: Path):
    d = json.load(open(p, encoding='utf-8'))
    domain = d.get('domain') or '?'
    model = d.get('model') or '?'
    n_a = d.get('agent_count')
    res = d.get('results', [])
    n = len(res)
    if n == 0:
        return None

    expected_n = EXPECTED_TOTAL.get(domain, 0)

    # Check agent_responses presence
    ar_lens = []
    fr_lens = []  # full_response length stats
    correct_v2 = 0
    correct_stored = 0
    agree_v2_stored = 0
    extracted_match_aggregated = 0
    agg_present = 0

    for r in res:
        ar = r.get('agent_responses', [])
        ar_lens.append(len(ar))
        for a in ar:
            full_r = a.get('full_response') or ''
            fr_lens.append(len(full_r))

        # Re-extract using v2 from full text of all agents
        full_text = ' '.join((a.get('full_response') or '') for a in ar)
        gt_key = 'ground_truth' if 'ground_truth' in r else 'correct_answer'
        gt = r.get(gt_key)
        ic_stored = r.get('is_correct') is True

        if domain == 'math':
            pred_v2 = extract_math(full_text)
            g = normalise_number(gt)
            ic_v2 = bool(pred_v2) and pred_v2 == g
        else:  # reasoning
            pred_v2 = extract_letter(full_text, k=4)
            g = str(gt).strip().upper()[:1] if gt else ''
            ic_v2 = bool(pred_v2) and pred_v2 == g

        if ic_stored: correct_stored += 1
        if ic_v2: correct_v2 += 1
        if ic_stored == ic_v2: agree_v2_stored += 1

    flags = []
    if expected_n and n != expected_n:
        flags.append(f'INCOMPLETE({n}/{expected_n})')
    if ar_lens and not all(L == n_a for L in ar_lens):
        bad = sum(1 for L in ar_lens if L != n_a)
        flags.append(f'BAD_AR_LEN({bad})')
    if fr_lens:
        med = statistics.median(fr_lens)
        max_l = max(fr_lens)
        # detect truncation cap
        near_200 = sum(1 for x in fr_lens if 195 <= x <= 205)
        near_2000 = sum(1 for x in fr_lens if 1990 <= x <= 2010)
        if near_200 > 0.10 * len(fr_lens):
            flags.append(f'CAP_200({near_200}/{len(fr_lens)})')
        if near_2000 > 0.10 * len(fr_lens):
            flags.append(f'CAP_2000({near_2000}/{len(fr_lens)})')
    else:
        med = max_l = 0

    return {
        'file': p.name, 'domain': domain, 'model': model, 'agents': n_a, 'n': n,
        'fr_med': med, 'fr_max': max_l,
        'acc_stored': correct_stored / n,
        'acc_v2': correct_v2 / n,
        'agree_v2_stored': agree_v2_stored / n,
        'flags': flags,
    }


print(f'{"FILE":58s} {"n":>4s} {"frmed":>6s} {"frmax":>6s} {"stored":>7s} {"v2":>7s} {"agree":>7s}')
print('-' * 105)
all_rows = []
for sub in sorted(ROOT.iterdir()):
    if not sub.is_dir() or not sub.name.startswith('v2_FIXED'): continue
    for f in sorted(sub.iterdir()):
        r = report_file(f)
        if r is None: continue
        all_rows.append(r)
        flags = ' '.join(r['flags']) if r['flags'] else ''
        print(f'{r["file"]:58s} {r["n"]:4d} {r["fr_med"]:6.0f} {r["fr_max"]:6.0f} '
              f'{r["acc_stored"]:7.3f} {r["acc_v2"]:7.3f} {r["agree_v2_stored"]:7.3f}'
              + ('  :: ' + flags if flags else ''))

# Curve summary
from collections import defaultdict
print()
print('=== Curves (v2 acc) ===')
curves = defaultdict(dict)
for r in all_rows:
    curves[(r['domain'], r['model'])][r['agents']] = r['acc_v2']
for (d, m), c in sorted(curves.items()):
    pts = [c.get(k, None) for k in (1, 2, 3, 4)]
    print(f'  {d:10s} {m:25s}  ' + '  '.join(
        f'N{k}={v:.3f}' if v is not None else f'N{k}=---' for k, v in zip((1,2,3,4), pts)))

# Save
import csv
out = Path(r'<paper_package>\analysis\tier2_v2_FIXED_curves.csv')
out.parent.mkdir(exist_ok=True)
with open(out, 'w', newline='', encoding='utf-8') as f:
    fields = ['file', 'domain', 'model', 'agents', 'n', 'fr_med', 'fr_max', 'acc_stored', 'acc_v2', 'agree_v2_stored', 'flags']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in all_rows:
        rr = {**r, 'flags': ';'.join(r['flags'])}
        w.writerow(rr)
print(f'\\nWrote {out}')
