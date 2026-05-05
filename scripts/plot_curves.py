"""핵심 figure 생성:
  Figure 1: agent 수 × accuracy 곡선 (5 domains × 2 models, contamination flag 표시)
  Figure 2: K vs observed peak agent count (단순 결정 규칙 검증)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'analysis'
FIG = ROOT / 'paper' / 'figures_v2'
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

EXCLUDED = {  # contaminated cells (T0.1 / T0.2)
    ('code', 'gpt_4o', 1), ('code', 'gpt_4o', 2), ('code', 'gpt_4o', 3), ('code', 'gpt_4o', 4),
    ('code', 'claude_sonnet_4_6', 1), ('code', 'claude_sonnet_4_6', 2),
    ('code', 'claude_sonnet_4_6', 3), ('code', 'claude_sonnet_4_6', 4),
    ('medical', 'claude_sonnet_4_6', 1),
    ('logic', 'gpt_4o', 4),
}

DOMAIN_COLORS = {
    'medical': '#d62728',
    'legal': '#1f77b4',
    'code': '#2ca02c',
    'logic': '#9467bd',
    'long_context': '#ff7f0e',
}

def fig1_curves():
    per_q = pd.read_csv(A / 'per_question.csv')
    feats = pd.read_csv(A / 'domain_features_combined.csv')
    K_by_dom = dict(zip(feats['domain'], feats['K']))

    # Compute mean accuracy per (dom, model, agents) excluding contaminated
    # Also compute bootstrap CI
    rng = np.random.default_rng(42)
    cells = []
    for (dom, model, a), sub in per_q.groupby(['domain', 'model', 'agent_count']):
        if (dom, model, a) in EXCLUDED:
            cells.append({'domain':dom,'model':model,'agents':a,'acc':None,'ci_lo':None,'ci_hi':None,'n':len(sub),'excluded':True})
            continue
        arr = sub['is_correct'].astype(int).to_numpy()
        if len(arr) < 100:  # too small
            cells.append({'domain':dom,'model':model,'agents':a,'acc':None,'ci_lo':None,'ci_hi':None,'n':len(arr),'excluded':True})
            continue
        boots = rng.choice(arr, size=(1000, len(arr)), replace=True).mean(axis=1)
        cells.append({
            'domain': dom, 'model': model, 'agents': a,
            'acc': float(arr.mean()), 'ci_lo': float(np.percentile(boots, 2.5)),
            'ci_hi': float(np.percentile(boots, 97.5)), 'n': len(arr), 'excluded': False,
        })
    cells_df = pd.DataFrame(cells)

    domains = ['medical', 'legal', 'logic', 'long_context', 'code']
    models = ['gpt_4o', 'claude_sonnet_4_6', 'gemini_2_5_pro']
    model_styles = {
        'gpt_4o': {'marker': 'o', 'linestyle': '-', 'label': 'GPT-4o'},
        'claude_sonnet_4_6': {'marker': 's', 'linestyle': '--', 'label': 'Claude Sonnet 4.6'},
        'gemini_2_5_pro': {'marker': '^', 'linestyle': ':', 'label': 'Gemini 2.5 Pro'},
    }

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True)
    for ax, dom in zip(axes, domains):
        K = K_by_dom.get(dom, None)
        sub_dom = cells_df[cells_df['domain'] == dom]
        for model in models:
            sub = sub_dom[(sub_dom['model'] == model) & (~sub_dom['excluded'])].sort_values('agents')
            if len(sub) >= 2:
                ax.plot(sub['agents'], sub['acc'] * 100, color=DOMAIN_COLORS[dom],
                        alpha=0.95, **model_styles[model])
                ax.fill_between(sub['agents'], sub['ci_lo']*100, sub['ci_hi']*100,
                                 color=DOMAIN_COLORS[dom], alpha=0.12)
            # excluded markers (greyed X)
            ex = sub_dom[(sub_dom['model'] == model) & (sub_dom['excluded'])]
            if len(ex):
                ax.scatter(ex['agents'], [ax.get_ylim()[1] * 0.5] * len(ex),
                           marker='x', color='lightgrey', s=40, alpha=0.5)

        title = f'{dom}\n(K={K:.2f})' if K is not None else dom
        ax.set_title(title)
        ax.set_xlabel('# agents')
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.3)
        if dom == domains[0]:
            ax.set_ylabel('Accuracy (%)')

    # legend
    handles = [plt.Line2D([0], [0], **model_styles[m], color='gray') for m in models]
    fig.legend(handles, [model_styles[m]['label'] for m in models],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Accuracy vs. agent count by task domain — observed curves\n'
                  '(Code domain excluded due to data contamination; medical/Claude/1-agent excluded due to n=150)',
                  fontsize=11)
    plt.tight_layout()
    out = FIG / 'fig1_curves_by_domain.pdf'
    out_png = FIG / 'fig1_curves_by_domain.png'
    plt.savefig(out)
    plt.savefig(out_png)
    plt.close()
    print(f'Wrote {out}')
    print(f'Wrote {out_png}')


def fig2_K_vs_peak():
    pred = pd.read_csv(A / 'predictive_model.csv')
    feats = pd.read_csv(A / 'domain_features_combined.csv')
    pred_eval = pred[~pred['shape_observed'].isin(['insufficient'])].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # observed peak vs K
    for _, r in pred_eval.iterrows():
        ax.scatter(r['K'], r['peak_observed'],
                   color=DOMAIN_COLORS[r['domain']],
                   marker='o' if r['model'] == 'gpt_4o' else 's',
                   s=120, edgecolor='black', alpha=0.85, zorder=3)
        ax.annotate(f"{r['domain']}\n({r['model'][:6]})",
                     (r['K'], r['peak_observed']),
                     fontsize=8, xytext=(8, 4), textcoords='offset points')
    # decision boundary
    ax.axvline(0.6, color='red', linestyle='--', alpha=0.6,
                label='Decision rule:\nK ≥ 0.6 → 1-agent peak')
    # predicted peak as background
    ax.fill_betweenx([0, 5], 0.6, 1.0, color='red', alpha=0.07)
    ax.fill_betweenx([0, 5], 0.0, 0.6, color='blue', alpha=0.07)
    ax.text(0.8, 4.5, 'predict 1-agent', color='red', alpha=0.6, ha='center')
    ax.text(0.3, 4.5, 'predict 2-agent', color='blue', alpha=0.6, ha='center')

    ax.set_xlabel('K (knowledge concentration, LLM-judge mean)')
    ax.set_ylabel('Observed peak agent count')
    ax.set_title('K predicts peak agent count\n(n=8 trustable cells, exact match=88%, MAE=0.38)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend(loc='center right', fontsize=9)
    plt.tight_layout()
    out = FIG / 'fig2_K_vs_peak.pdf'
    plt.savefig(out)
    plt.savefig(FIG / 'fig2_K_vs_peak.png')
    plt.close()
    print(f'Wrote {out}')


def fig3_features_table():
    feats = pd.read_csv(A / 'domain_features_combined.csv')
    feats = feats.sort_values('K', ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    domains = feats['domain'].tolist()
    axes_lbl = ['D', 'V', 'K', 'S']
    M = feats[axes_lbl].to_numpy()
    im = ax.imshow(M.T, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=15)
    ax.set_yticks(range(len(axes_lbl)))
    ax.set_yticklabels(axes_lbl)
    for i in range(len(domains)):
        for j, ax_lbl in enumerate(axes_lbl):
            ax.text(i, j, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=10,
                    color='black' if 0.3 < M[i, j] < 0.7 else 'white')
    plt.colorbar(im, ax=ax, label='value [0,1]')
    ax.set_title('Task feature scores per domain (mean of GPT-4o + Gemini 2.5 Pro judges, 3 reps each)')
    plt.tight_layout()
    out = FIG / 'fig3_features_heatmap.pdf'
    plt.savefig(out); plt.savefig(FIG / 'fig3_features_heatmap.png')
    plt.close()
    print(f'Wrote {out}')


def main():
    fig1_curves()
    fig2_K_vs_peak()
    fig3_features_table()

if __name__ == '__main__':
    main()
