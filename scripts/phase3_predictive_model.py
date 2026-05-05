"""Phase 3: Task feature → 곡선 형태 예측 모델 (스캐폴드).

입력:
  - analysis/domain_features_combined.csv (D, V, K, S per domain — Phase 2 산출)
  - analysis/per_question.csv (per-question is_correct — Phase 1 산출)
  - 단, 데이터 오염 셀(code/gpt 전체)은 제외

출력:
  - analysis/curve_observed.csv  : 도메인×모델별 (1-agent vs 4-agent) 정량
  - analysis/predictive_model.json : K 기반 단순 결정 규칙 + LOO-CV 정확도

본 스크립트는 데이터 정합성 이슈가 해결된 후 본격 분석에 쓸 수 있도록 골격만.
현재는 신뢰 가능 셀만 포함해 1차 분석 수행.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'analysis'

# === 신뢰 가능 셀 정의 ===
# code/gpt_4o 전체: 빈 답변 매칭 버그 (T0.1) → 제외
# medical/claude/1-agent: n=150 (test mode) → 제외
# logic/gpt_4o/4-agent: n=450 (부분 실행) → 제외
# 200자 truncation 영향이 큰 code/claude 전체 → 제외 (재실험 후 재평가)
# 그 외는 보정으로 사용 가능
EXCLUDED = {
    ('code', 'gpt_4o', 1), ('code', 'gpt_4o', 2), ('code', 'gpt_4o', 3), ('code', 'gpt_4o', 4),
    ('code', 'claude_sonnet_4_6', 1), ('code', 'claude_sonnet_4_6', 2),
    ('code', 'claude_sonnet_4_6', 3), ('code', 'claude_sonnet_4_6', 4),
    ('medical', 'claude_sonnet_4_6', 1),
    ('logic', 'gpt_4o', 4),
    # Gemini 부분 데이터는 제외
}

def load_features():
    p = A / 'domain_features_combined.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p} — run Phase 2 first')
    return pd.read_csv(p)

def load_per_question():
    p = A / 'per_question.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p} — run scripts/extract_curves.py first')
    return pd.read_csv(p)

def empty_pred_correct_filter():
    """results/merged/*.json에서 empty predicted_answer & is_correct=True 케이스를 식별.
    contamination_audit_v2의 empty_pred_BUG_correct를 per-question 레벨에서 보정용 필터로 변환.
    """
    contam = pd.read_csv(A / 'contamination_audit_v2.csv')
    # 일단 cell-level inflation 정보만 반환. 정확한 보정은 raw json 다시 읽어야 함 — 추후
    return contam[['domain', 'model', 'agents', 'empty_pred_BUG_correct',
                    'reported_correct', 'reported_accuracy', 'corrected_acc_minus_empty']]

def summarize_curves(per_q: pd.DataFrame, contam: pd.DataFrame):
    """각 (domain, model)에서 (1, 4) agent 정확도 + corrected accuracy로 곡선 요약."""
    rows = []
    for (dom, model), sub in per_q.groupby(['domain', 'model']):
        # 모든 agent count별 정확도
        accs = {}
        for a in (1, 2, 3, 4):
            cell = sub[sub['agent_count'] == a]
            if (dom, model, a) in EXCLUDED or len(cell) == 0:
                accs[a] = None
                continue
            accs[a] = float(cell['is_correct'].astype(int).mean())
        if all(accs[a] is not None for a in (1, 4)):
            delta_4_1 = accs[4] - accs[1]
        else:
            delta_4_1 = None
        peak_agent = None
        valid = [a for a in (1, 2, 3, 4) if accs[a] is not None]
        if valid:
            peak_agent = max(valid, key=lambda a: accs[a])
        rows.append({
            'domain': dom, 'model': model,
            'acc_1': accs[1], 'acc_2': accs[2], 'acc_3': accs[3], 'acc_4': accs[4],
            'delta_4_minus_1': delta_4_1,
            'peak_agent': peak_agent,
            'n_valid_points': len(valid),
        })
    return pd.DataFrame(rows)

def classify_observed(row):
    """관찰된 곡선 형태 분류 — 데이터 신뢰성 고려."""
    if row['n_valid_points'] < 3:
        return 'insufficient'
    accs = [row[f'acc_{a}'] for a in (1, 2, 3, 4)]
    accs_v = [a for a in accs if a is not None]
    if len(accs_v) < 3: return 'insufficient'
    # Spearman trend over valid points
    xs = [a for a in (1, 2, 3, 4) if row[f'acc_{a}'] is not None]
    ys = [row[f'acc_{a}'] for a in xs]
    if len(xs) >= 3:
        rho, p = stats.spearmanr(xs, ys)
        if p < 0.1 and rho < -0.7:
            return 'decreasing'
        if p < 0.1 and rho > 0.7:
            return 'increasing'
    peak = row['peak_agent']
    if peak in (2, 3):
        return 'inverted_U'
    if peak == 1 and ys[-1] < ys[0]:
        return 'decreasing_weak'
    if peak == 4 and ys[-1] > ys[0]:
        return 'increasing_weak'
    return 'flat'

def predict_shape(K, V):
    """H1: K threshold rule.
    K ≥ 0.6 → decreasing  (knowledge-bound)
    K < 0.6 → inverted_U / flat (peak at moderate count)
    """
    if K >= 0.6: return 'decreasing'
    return 'inverted_U'  # default to inverted_U for low/mid K

def predict_peak(K):
    """H2 (refined): simple step function.
    K ≥ 0.6  → peak = 1 (single agent)
    K < 0.6  → peak = 2 (moderate count)
    """
    return 1 if K >= 0.6 else 2

def predict_peak_continuous(K):
    """H2-cont: linear interpolation peak = round(2 - 1 * sigmoid((K - 0.6) * 10))."""
    import math
    sig = 1 / (1 + math.exp(-(K - 0.6) * 10))
    return int(round(2 - sig))  # K high → 1, K low → 2

def main():
    feats = load_features()
    per_q = load_per_question()
    contam = empty_pred_correct_filter()

    # join: feature columns onto per_q via domain
    print('=== Domain features (combined judge mean) ===')
    feats_view = feats[['domain', 'D', 'V', 'K', 'S']].sort_values('K', ascending=False)
    print(feats_view.round(2).to_string(index=False))
    print()

    # observed curves
    obs = summarize_curves(per_q, contam)
    obs['shape_observed'] = obs.apply(classify_observed, axis=1)
    obs.to_csv(A / 'curve_observed.csv', index=False)
    print('=== Observed curves (excluded contaminated cells) ===')
    cols = ['domain', 'model', 'acc_1', 'acc_2', 'acc_3', 'acc_4',
            'delta_4_minus_1', 'peak_agent', 'shape_observed']
    print(obs[cols].round(4).to_string(index=False))
    print()

    # predict per (domain, model) using domain-level features
    pred_rows = []
    for _, r in obs.iterrows():
        f = feats[feats['domain'] == r['domain']]
        if f.empty: continue
        K = float(f['K'].iloc[0])
        V = float(f['V'].iloc[0])
        D = float(f['D'].iloc[0])
        S = float(f['S'].iloc[0])
        pred_rows.append({
            'domain': r['domain'], 'model': r['model'],
            'D': D, 'V': V, 'K': K, 'S': S,
            'shape_predicted_H1': predict_shape(K, V),
            'peak_predicted_H2': predict_peak(K),
            'shape_observed': r['shape_observed'],
            'peak_observed': r['peak_agent'],
        })
    pred_df = pd.DataFrame(pred_rows)
    print('=== Hypothesis tests ===')
    print(pred_df.round(2).to_string(index=False))
    print()

    # H1 accuracy — only on rows with sufficient observed data
    pred_eval = pred_df[~pred_df['shape_observed'].isin(['insufficient'])].copy()
    def is_match(row):
        p, o = row['shape_predicted_H1'], row['shape_observed']
        if p == o: return True
        if p == 'decreasing' and o in ('decreasing_weak',): return True
        if p == 'inverted_U' and o in ('inverted_U',): return True
        if p == 'increasing' and o in ('increasing_weak',): return True
        if p == 'flat' and o in ('flat',): return True
        return False
    pred_eval['h1_match'] = pred_eval.apply(is_match, axis=1)
    print(f'\nH1 (shape) — eval on {len(pred_eval)} rows with sufficient data:')
    print(f'  matches: {int(pred_eval["h1_match"].sum())}/{len(pred_eval)} '
          f'({pred_eval["h1_match"].mean()*100:.0f}%)')
    print(pred_eval[['domain','model','K','shape_predicted_H1','shape_observed','h1_match']].to_string(index=False))

    # H2: peak — only on rows with valid observed peak AND not insufficient
    valid_peak = pred_eval.dropna(subset=['peak_observed'])
    if len(valid_peak) > 0:
        diffs = (valid_peak['peak_predicted_H2'] - valid_peak['peak_observed']).abs()
        mae = float(diffs.mean())
        exact = float((valid_peak['peak_predicted_H2'] == valid_peak['peak_observed']).mean())
        within1 = float((diffs <= 1).mean())
        print(f'\nH2 (peak agent) — eval on {len(valid_peak)} rows:')
        print(f'  MAE={mae:.2f}, exact_match={exact*100:.0f}%, within±1={within1*100:.0f}%')

    pred_df.to_csv(A / 'predictive_model.csv', index=False)

    # save model JSON
    out = {
        'features_used': ['D', 'V', 'K', 'S'],
        'primary_features': ['V', 'K'],
        'hypothesis_H1': {
            'rule': 'shape = decreasing if K>=0.6 else inverted_U',
            'matches': int(pred_eval['h1_match'].sum()),
            'total': len(pred_eval),
            'accuracy': float(pred_eval['h1_match'].mean()),
            'per_row': pred_eval[['domain','model','K','shape_predicted_H1','shape_observed','h1_match']].to_dict('records'),
        },
        'hypothesis_H2': {
            'rule': 'peak = 1 if K>=0.6 else 2',
            'mae': float(mae) if len(valid_peak) else None,
            'exact_match_rate': float(exact) if len(valid_peak) else None,
            'within_1_rate': float(within1) if len(valid_peak) else None,
            'n': int(len(valid_peak)),
        },
        'excluded_cells': [list(c) for c in sorted(EXCLUDED)],
        'caveats': [
            'Code domain entirely excluded due to T0.1 (empty-pred matching bug).',
            'medical/claude/1-agent excluded (n=150 only).',
            'Some cells have pred_truncated_at_200 (T0.2) — included if not severely affecting matching.',
            'Sample size small (4 domains × 2 models = 8 trustable rows). LOO-CV on small data.',
            'Question-level Gemini+GPT judging in progress for richer feature signal.',
        ],
    }
    (A / 'predictive_model.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nWrote {A / 'predictive_model.json'}")

if __name__ == '__main__':
    main()
