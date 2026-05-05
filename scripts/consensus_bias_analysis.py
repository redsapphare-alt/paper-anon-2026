"""Consensus bias analysis — does majority vote at high N collapse onto a modal answer?

Hypothesis: at high N (4 agents), majority vote amplifies model bias toward the modal
answer choice, producing accuracy near random when the modal answer ≠ ground truth.

Method:
  For each (domain, model) cell at N=1 and N=4, compute:
    - Distribution of predictions across answer choices
    - Modal-answer concentration (max share)
    - Gini coefficient
    - Entropy

Predictions:
  - N=1: more diverse predictions (lower modal concentration)
  - N=4: vote collapses to mode (higher modal concentration)

Output:
  analysis/consensus_bias.csv
  paper/latex_v2/figures/fig9_consensus_bias.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'results' / 'merged'
A = ROOT / 'analysis'
FIG = ROOT / 'paper' / 'latex_v2' / 'figures'

matplotlib.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                              'font.family': 'sans-serif',
                              'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})

DOMAIN_COLORS = {'medical': '#d62728', 'legal': '#1f77b4',
                 'logic': '#9467bd', 'long_context': '#ff7f0e'}

# Trustable cells: (domain, model, file_pattern)
CELLS = [
    ('legal', 'gpt_4o',            'legal_gpt_4o_{}agent_parallel.json'),
    ('legal', 'claude_sonnet_4_6', 'legal_claude_sonnet_4_6_{}agent.json'),
    ('medical', 'gpt_4o',          'medical_gpt_4o_{}agent.json'),
    ('logic', 'gpt_4o',            'logic_gpt_4o_{}agent.json'),
    ('logic', 'claude_sonnet_4_6', 'logic_claude_sonnet_4_6_{}agent.json'),
    ('long_context', 'gpt_4o',     'long_context_gpt_4o_{}agent_parallel.json'),
    ('long_context', 'claude_sonnet_4_6', 'long_context_claude_sonnet_4_6_{}agent.json'),
]


import re

ANSWER_RE = re.compile(r'(?:answer|final answer|the answer is)[\s:]*([A-Za-z0-9])',
                        re.IGNORECASE)


def normalise_pred(s: str) -> str:
    """Extract the chosen answer token. Tries 'Answer: X' first, then first alnum char."""
    if not s: return '__empty__'
    s = s.strip()
    if not s: return '__empty__'
    # Try "Answer: A" pattern
    m = ANSWER_RE.search(s)
    if m:
        return m.group(1).lower()
    # Fallback: first single letter/digit token
    s_low = s.lower()
    # Look for standalone single-letter tokens (often the model just outputs "A")
    if len(s_low) <= 3:
        return s_low.strip(' .,:;')[:1] or '__empty__'
    # Look for "(A)" or "A)" or "A." patterns
    m2 = re.search(r'\b([A-Da-d])\b', s)
    if m2:
        return m2.group(1).lower()
    # Last resort
    first_token = re.match(r'[A-Za-z0-9]+', s_low)
    return first_token.group(0)[:5] if first_token else '__empty__'


def cell_pred_dist(domain, model, agents):
    fname_template = next((tpl for d, m, tpl in CELLS if d == domain and m == model), None)
    if fname_template is None: return None
    p = M / fname_template.format(agents)
    if not p.exists(): return None
    d = json.load(open(p, encoding='utf-8'))
    res = d.get('results', [])
    if len(res) < 100: return None
    preds = [normalise_pred(r.get('predicted_answer') or '') for r in res]
    counts = Counter(preds)
    n = len(preds)
    return {'n': n, 'counts': counts, 'top_pred': counts.most_common(1)[0][0],
            'top_share': counts.most_common(1)[0][1] / n,
            'entropy': sum(-c/n * np.log(c/n) for c in counts.values() if c > 0)}


# Compute for all trustable cells × (N=1, N=4)
records = []
for dom, model, _ in CELLS:
    for n_agents in (1, 4):
        info = cell_pred_dist(dom, model, n_agents)
        if info is None:
            print(f'  skip {dom}/{model}/N={n_agents} (missing)')
            continue
        records.append({'domain': dom, 'model': model, 'N': n_agents,
                        'n_questions': info['n'],
                        'modal_share': info['top_share'],
                        'modal_pred': info['top_pred'],
                        'entropy': info['entropy'],
                        'top5': info['counts'].most_common(5)})

df = pd.DataFrame(records)
print('=== Modal-answer concentration ===')
for (dom, model), sub in df.groupby(['domain', 'model']):
    if len(sub) != 2: continue
    s1 = sub[sub.N == 1].iloc[0]
    s4 = sub[sub.N == 4].iloc[0]
    print(f'  {dom:14s} {model:18s}: N=1 modal={s1["modal_pred"]} '
          f'({s1["modal_share"]:.2f})  →  N=4 modal={s4["modal_pred"]} '
          f'({s4["modal_share"]:.2f})  Δ={s4["modal_share"]-s1["modal_share"]:+.2f}')

# Save for paper
out_df = df[['domain', 'model', 'N', 'n_questions', 'modal_share',
              'modal_pred', 'entropy']].copy()
out_df.to_csv(A / 'consensus_bias.csv', index=False)
print(f'\nWrote {A / "consensus_bias.csv"}')

# =====================================================================
# Figure: stacked bar of prediction distributions per cell, N=1 vs N=4
# =====================================================================
trustable_pairs = [(d, m) for d, m, _ in CELLS]
# Drop pairs missing one of the two N values
valid_pairs = []
for d, m in trustable_pairs:
    sub = df[(df.domain == d) & (df.model == m)]
    if len(sub) == 2: valid_pairs.append((d, m))

n_pairs = len(valid_pairs)
fig, axes = plt.subplots(2, n_pairs, figsize=(2 * n_pairs, 5.5),
                           sharey='row')
if n_pairs == 1:
    axes = axes[:, None]

# Top row: prediction distribution per cell
all_token_set = set()
for d, m in valid_pairs:
    fname_template = next(tpl for dd, mm, tpl in CELLS if dd == d and mm == m)
    for n_a in (1, 4):
        p = M / fname_template.format(n_a)
        if p.exists():
            data = json.load(open(p, encoding='utf-8'))
            for r in data.get('results', [])[:5000]:
                all_token_set.add(normalise_pred(r.get('predicted_answer') or ''))

# Restrict to top tokens for clarity
common_tokens = ['a', 'b', 'c', 'd', 'yes', 'no', 'rele', 'irre', 'true', 'fals', '__empty__']
plot_tokens = [t for t in common_tokens if t in all_token_set] + ['other']

palette = plt.cm.tab20(np.linspace(0, 1, len(plot_tokens)))
token_color = dict(zip(plot_tokens, palette))


def get_dist_array(domain, model, n_agents):
    fname_template = next(tpl for dd, mm, tpl in CELLS if dd == domain and mm == model)
    p = M / fname_template.format(n_agents)
    if not p.exists(): return None
    data = json.load(open(p, encoding='utf-8'))
    counts = Counter()
    for r in data.get('results', []):
        tok = normalise_pred(r.get('predicted_answer') or '')
        if tok in plot_tokens:
            counts[tok] += 1
        else:
            counts['other'] += 1
    n = sum(counts.values())
    if n == 0: return None
    return {t: counts.get(t, 0) / n for t in plot_tokens}


for i, (d, m) in enumerate(valid_pairs):
    label = f'{d[:6]}/{"GPT" if m == "gpt_4o" else "Cl"}'
    for j, n_a in enumerate([1, 4]):
        ax = axes[j, i]
        dist = get_dist_array(d, m, n_a)
        if dist is None:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        bottom = 0
        for tok in plot_tokens:
            v = dist.get(tok, 0)
            if v > 0:
                ax.bar([0], [v], bottom=[bottom], color=token_color[tok],
                       edgecolor='white', linewidth=0.4, label=tok if (i == 0 and j == 0) else None)
                if v > 0.05:
                    ax.text(0, bottom + v / 2, tok, ha='center', va='center', fontsize=7,
                            color='white' if v > 0.20 else 'black')
                bottom += v
        ax.set_ylim(0, 1.02)
        ax.set_xticks([])
        if j == 0:
            ax.set_title(label, fontsize=9)
        if i == 0:
            ax.set_ylabel(f'N={n_a}\nshare')
        # Annotate modal share
        sub = df[(df.domain == d) & (df.model == m) & (df.N == n_a)]
        if len(sub):
            modal = sub.iloc[0]
            ax.text(0.5, 1.04, f'{modal["modal_pred"]}: {modal["modal_share"]*100:.0f}%',
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=7,
                    fontweight='bold')

fig.suptitle('Consensus bias: prediction distribution at N=1 (top) vs N=4 (bottom)',
              y=0.99, fontsize=11, fontweight='bold')
fig.text(0.5, 0.02,
          'Modal-answer share rises sharply at N=4 on knowledge-bound cells — '
          'voting amplifies the model\'s biased mode.',
          ha='center', va='bottom', fontsize=9, fontstyle='italic', color='#444')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
out = FIG / 'fig9_consensus_bias.pdf'
plt.savefig(out, bbox_inches='tight')
plt.savefig(FIG / 'fig9_consensus_bias.png', bbox_inches='tight')
plt.close()
print(f'Wrote {out}')
