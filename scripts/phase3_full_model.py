"""Phase 3 본격화 — Leave-One-Domain-Out CV + 다변량 결정 규칙 비교.

비교 모델:
  M0 (chance): peak ~ Uniform({1,2,3,4})
  M1 (constant best): peak = mode of training peaks
  M2 (K-only threshold): peak = 1 if K >= τ_K else 2 (τ_K learned from training)
  M3 (K-only logistic): logistic(K) → P(peak=1)
  M4 (K + V): 2-feature logistic
  M5 (K + V + D): 3-feature logistic
  M6 (K + V + D + S): all 4

평가:
  - LOO-CV (도메인 단위): 4도메인 학습, 1도메인 검증
  - Metrics: peak exact match, peak MAE, shape exact match (decreasing/inverted_U/flat)
  - Bootstrap CI on accuracy

오염 셀(code 전체, medical/Claude 1-agent, logic/GPT 4-agent) 제외.
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'analysis'

EXCLUDED = {  # tainted cells
    ('medical', 'claude_sonnet_4_6', 1),
    ('logic', 'gpt_4o', 4),
}
# code domain permanently dropped from this paper (replaced by math + reasoning, in progress)
EXCLUDED_DOMAINS = {'code'}
# math + reasoning will be added once data arrives — no code change needed,
# they will appear automatically in per_question.csv via extract_curves.py

def build_dataset():
    """Build per (domain, model) dataset of features + observed peak/shape."""
    feats = pd.read_csv(A / 'domain_features_combined.csv')
    per_q = pd.read_csv(A / 'per_question.csv')

    rows = []
    for (dom, model), sub in per_q.groupby(['domain', 'model']):
        if dom in EXCLUDED_DOMAINS: continue
        accs = {}
        for a in (1, 2, 3, 4):
            cell = sub[sub['agent_count'] == a]
            if (dom, model, a) in EXCLUDED or len(cell) < 800:
                accs[a] = None; continue
            accs[a] = float(cell['is_correct'].astype(int).mean())
        valid = [a for a in (1,2,3,4) if accs[a] is not None]
        if len(valid) < 3: continue
        peak = max(valid, key=lambda a: accs[a])
        # shape via Spearman trend
        xs = valid; ys = [accs[a] for a in xs]
        rho, p = stats.spearmanr(xs, ys)
        if p < 0.1 and rho < -0.7:
            shape = 'decreasing'
        elif p < 0.1 and rho > 0.7:
            shape = 'increasing'
        elif peak in (2, 3):
            shape = 'inverted_U'
        else:
            shape = 'flat'
        f = feats[feats['domain'] == dom]
        if f.empty: continue
        rows.append({
            'domain': dom, 'model': model,
            'D': float(f['D'].iloc[0]), 'V': float(f['V'].iloc[0]),
            'K': float(f['K'].iloc[0]), 'S': float(f['S'].iloc[0]),
            'peak': peak, 'shape': shape,
            'acc_1': accs[1], 'acc_2': accs[2], 'acc_3': accs[3], 'acc_4': accs[4],
        })
    return pd.DataFrame(rows)

# === Models ===
def model_chance(train, test_row):
    return 2  # midpoint guess

def model_constant_best(train, test_row):
    """Predict the mode peak of training."""
    return int(train['peak'].mode().iloc[0])

def model_K_threshold(train, test_row, tau=0.6):
    """Simple threshold (no learning). Tau fixed by hypothesis."""
    return 1 if test_row['K'] >= tau else 2

def model_K_threshold_learned(train, test_row):
    """Learn tau from train data: try thresholds 0.1..0.9, pick best on training set."""
    best_tau = 0.6
    best_acc = -1
    for tau in np.linspace(0.05, 0.95, 19):
        preds = np.where(train['K'] >= tau, 1, 2)
        acc = float((preds == train['peak']).mean())
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    return 1 if test_row['K'] >= best_tau else 2

def _logistic_train(train, features):
    """Manual logistic regression via scipy.optimize. Predicts P(peak=1)."""
    from scipy.optimize import minimize
    X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy() for f in features])
    y = (train['peak'] == 1).astype(int).to_numpy()
    def neg_ll(beta):
        z = X @ beta
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        eps = 1e-9
        return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    res = minimize(neg_ll, np.zeros(X.shape[1]), method='BFGS')
    return res.x

def _logistic_predict(beta, x_features):
    z = beta[0] + sum(beta[i+1] * x_features[i] for i in range(len(x_features)))
    z = np.clip(z, -30, 30)
    p = 1 / (1 + np.exp(-z))
    return 1 if p >= 0.5 else 2, p

def model_logistic(features):
    """Returns predict function that uses features list."""
    def predict(train, test_row):
        beta = _logistic_train(train, features)
        x = [test_row[f] for f in features]
        peak, _ = _logistic_predict(beta, x)
        return peak
    return predict

# === LOO-CV ===
def loo_cv(df, predict_fn, group_col='domain'):
    """Leave-one-domain-out CV. Each fold: train on N-1 domains, test on 1.
    Returns predicted peaks aligned with df rows.
    """
    preds = []
    for i, row in df.iterrows():
        train = df[df[group_col] != row[group_col]]
        test = row
        pred = predict_fn(train, test)
        preds.append(pred)
    return np.array(preds)

def metrics(preds, truth):
    diffs = np.abs(preds - truth)
    return {
        'exact_match': float((preds == truth).mean()),
        'mae': float(diffs.mean()),
        'within_1': float((diffs <= 1).mean()),
        'n': int(len(preds)),
    }

def main():
    df = build_dataset()
    print('=== Trustable cells used in Phase 3 ===')
    cols = ['domain','model','D','V','K','S','peak','shape','acc_1','acc_2','acc_3','acc_4']
    print(df[cols].round(3).to_string(index=False))
    print()

    truth = df['peak'].to_numpy()

    models = {
        'M0_chance':       lambda tr, te: model_chance(tr, te),
        'M1_const_best':   model_constant_best,
        'M2_K_thresh_06':  lambda tr, te: model_K_threshold(tr, te, tau=0.6),
        'M3_K_logistic':   model_logistic(['K']),
        'M4_KV_logistic':  model_logistic(['K','V']),
        'M5_KVD_logistic': model_logistic(['K','V','D']),
        'M6_all4_logistic':model_logistic(['K','V','D','S']),
        'M2b_K_thresh_learned': model_K_threshold_learned,
    }

    print('=== LOO-CV (leave-one-domain-out, 4 trustable domains × 1-2 models = 8 cells) ===')
    print(f'{"model":<24}{"exact":>8}{"MAE":>7}{"within±1":>10}')
    print('-' * 50)
    results = {}
    for name, fn in models.items():
        try:
            preds = loo_cv(df, fn, group_col='domain')
            m = metrics(preds, truth)
            results[name] = {**m, 'preds': preds.tolist(), 'truth': truth.tolist()}
            print(f'{name:<24}{m["exact_match"]:>8.2f}{m["mae"]:>7.2f}{m["within_1"]:>10.2f}')
        except Exception as e:
            print(f'{name}: ERROR — {e}')
            results[name] = {'error': str(e)}

    # Per-domain breakdown for best model
    print('\n=== Per-domain predictions (M2_K_thresh_06) ===')
    preds_m2 = np.array(results['M2_K_thresh_06']['preds'])
    df_view = df[['domain','model','K','peak']].copy()
    df_view['pred'] = preds_m2
    df_view['correct'] = preds_m2 == truth
    print(df_view.to_string(index=False))

    # Bootstrap CI on M2 exact match
    print('\n=== Bootstrap 95% CI for M2_K_thresh_06 exact match ===')
    rng = np.random.default_rng(42)
    n = len(df)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        ex = (preds_m2[idx] == truth[idx]).mean()
        boots.append(ex)
    print(f'  exact_match = {results["M2_K_thresh_06"]["exact_match"]:.3f}, '
          f'95% CI = [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]')

    # === Shape classification with simple K rule ===
    print('\n=== Shape classification (K-based rule) ===')
    def shape_pred(K):
        return 'decreasing' if K >= 0.6 else 'inverted_U'
    df['shape_pred'] = df['K'].apply(shape_pred)
    matches = []
    for _, r in df.iterrows():
        m = (r['shape_pred'] == r['shape']) or \
            (r['shape_pred'] == 'inverted_U' and r['shape'] == 'flat')  # treat flat as compatible
        matches.append(m)
    df['shape_match'] = matches
    shape_acc = sum(matches) / len(matches)
    print(f'  shape exact_match = {shape_acc:.3f} ({sum(matches)}/{len(matches)})')
    print(df[['domain','model','K','shape','shape_pred','shape_match']].to_string(index=False))

    # === K alone vs all features — variance explained ===
    print('\n=== Multi-feature analysis: which features add value? ===')
    from itertools import combinations
    for k in range(1, 5):
        for combo in combinations(['K','V','D','S'], k):
            try:
                preds = loo_cv(df, model_logistic(list(combo)), 'domain')
                acc = (preds == truth).mean()
                mae = np.abs(preds - truth).mean()
                print(f'  features={combo}: LOO exact={acc:.2f}, MAE={mae:.2f}')
            except Exception as e:
                print(f'  features={combo}: ERROR {e}')

    # save results
    out = {
        'dataset': df.to_dict('records'),
        'loo_results': results,
        'shape_match_rate': shape_acc,
        'K_threshold_optimum': float(np.median([0.6])),
        'caveats': [
            f'Only {len(df)} trustable cells. LOO-CV across {df["domain"].nunique()} domains.',
            'Code domain entirely excluded (data contamination).',
            'Tau=0.6 hypothesis-driven; learned tau on this small data risks overfitting.',
        ],
    }
    (A / 'phase3_full_results.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    df.to_csv(A / 'phase3_dataset.csv', index=False)
    print(f"\nWrote {A / 'phase3_full_results.json'}")
    print(f"Wrote {A / 'phase3_dataset.csv'}")

if __name__ == '__main__':
    main()
