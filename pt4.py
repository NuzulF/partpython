############################################################
#                T6b_v3 — PART 4
#             FULL MODEL EVALUATION
############################################################

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, f1_score
)
import os
from pathlib import Path

current_dir = os.getcwd()

OUT_DIR = os.path.join(current_dir,"T6b_v3_FULL")

pred_seen_T6b = pd.read_csv(f"{OUT_DIR}/pred_seen_T6b_v3.csv")
unseen_matrix = pd.read_csv(f"{OUT_DIR}/pred_unseen_T6b_v3.csv", index_col=0)
test_df       = pd.read_csv(f"{OUT_DIR}/test_df_T6b_v3.csv")

print("Loaded evaluation inputs.")



############################################################
# 1. EVALUASI REGRESI UNTUK SEEN (Actual vs Pred_clipped)
############################################################
def evaluate_ratings_T6b(df_pred,
                         user_col='id_reviewer',
                         item_col='id_dtw',
                         rating_col='Actual',
                         pred_col='Pred_clipped',
                         rel_threshold=4.0):

    df_valid = df_pred.copy()
    df_valid = df_valid[df_valid[pred_col].notna() & df_valid[rating_col].notna()]

    if len(df_valid) == 0:
        return {k: np.nan for k in [
            'RMSE','MAE','MAPE_%','R2','EVS','Coverage','F1'
        ]}

    y_true = df_valid[rating_col].astype(float).to_numpy()
    y_pred = df_valid[pred_col].astype(float).to_numpy()

    err = y_pred - y_true

    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100)

    r2   = float(r2_score(y_true, y_pred)) if len(y_true)>=2 else np.nan
    evs  = float(explained_variance_score(y_true, y_pred)) if len(y_true)>=2 else np.nan

    y_true_bin = (y_true >= rel_threshold).astype(int)
    y_pred_bin = (y_pred >= rel_threshold).astype(int)

    f1 = float(f1_score(y_true_bin, y_pred_bin)) if len(np.unique(y_true_bin))>1 else np.nan

    coverage = len(df_valid) / len(df_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE_%': mape,
        'R2': r2,
        'EVS': evs,
        'Coverage': coverage,
        'F1': f1
    }



############################################################
# 2. TOP-N EVALUATION UNTUK UNSEEN
############################################################
def precision_recall_ndcg_at_k(unseen_matrix, test_df,
                               user_col='id_reviewer',
                               item_col='id_dtw',
                               rating_col='rating',
                               K=5,
                               rel_threshold=4.0):

    # Ground truth relevan
    gt_pos = (
        test_df[test_df[rating_col] >= rel_threshold]
        .groupby(user_col)[item_col]
        .apply(set)
        .to_dict()
    )

    total_hits = 0
    total_rec  = 0
    total_pos  = 0
    ndcgs = []

    for u in unseen_matrix.index:
        scores = unseen_matrix.loc[u].dropna()
        if scores.empty:
            continue

        topk = scores.sort_values(ascending=False).head(K).index.tolist()
        pos  = gt_pos.get(u, set())

        hits = len([it for it in topk if it in pos])
        total_hits += hits
        total_rec  += len(topk)
        total_pos  += len(pos)

        if len(topk) > 0 and len(pos) > 0:
            gains = [1 if it in pos else 0 for it in topk]
            dcg   = sum(gains[i] / np.log2(i + 2) for i in range(len(gains)))

            ideal = [1] * min(len(pos), K)
            idcg  = sum(ideal[i] / np.log2(i + 2) for i in range(len(ideal)))

            ndcgs.append(dcg / idcg if idcg > 0 else 0)

    precision = total_hits / total_rec if total_rec > 0 else np.nan
    recall    = total_hits / total_pos if total_pos > 0 else np.nan
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else np.nan

    return {
        f"Precision@{K}": precision,
        f"Recall@{K}": recall,
        f"NDCG@{K}": mean_ndcg
    }



############################################################
# 3. RANKING SEEN PREDICTION (Pred_clipped)
############################################################
def ranking_at_k_from_pred(df_pred,
                           user_col='id_reviewer',
                           item_col='id_dtw',
                           rating_col='Actual',
                           pred_col='Pred_clipped',
                           K=5,
                           rel_threshold=4.0):

    df_valid = df_pred.dropna(subset=[pred_col, rating_col])

    total_hits = 0
    total_rec  = 0
    total_pos  = 0
    ndcgs = []

    for u, g in df_valid.groupby(user_col):
        pos_items = set(g.loc[g[rating_col]>=rel_threshold, item_col].tolist())
        if len(pos_items) == 0:
            continue

        topk = g.sort_values(pred_col, ascending=False).head(K)[item_col].tolist()

        hits = len([it for it in topk if it in pos_items])
        total_hits += hits
        total_rec  += len(topk)
        total_pos  += len(pos_items)

        gains = [1 if it in pos_items else 0 for it in topk]
        dcg  = sum(gains[i]/np.log2(i+2) for i in range(len(gains)))
        idcg = sum(1/np.log2(i+2) for i in range(min(len(pos_items), K)))

        ndcgs.append(dcg/idcg if idcg > 0 else 0)

    precision = total_hits/total_rec if total_rec > 0 else np.nan
    recall    = total_hits/total_pos if total_pos > 0 else np.nan
    mean_ndcg = np.mean(ndcgs) if ndcgs else np.nan

    return {
        f"Precision@{K}": precision,
        f"Recall@{K}": recall,
        f"NDCG@{K}": mean_ndcg
    }



############################################################
# 4. RUN SEMUA EVALUASI
############################################################
eval_reg   = evaluate_ratings_T6b(pred_seen_T6b)
eval_unseen = precision_recall_ndcg_at_k(unseen_matrix, test_df, K=5)
eval_seen_rank = ranking_at_k_from_pred(pred_seen_T6b, K=5)


############################################################
# 5. SUMMARY & SIMPAN FILE
############################################################
summary_T6b_v3 = {
    **{f"rating_{k}": v for k, v in eval_reg.items()},
    **{f"unseen_{k}": v for k, v in eval_unseen.items()},
    **{f"test_{k}": v for k, v in eval_seen_rank.items()},
}

pd.DataFrame([summary_T6b_v3]).to_csv(
    f"{OUT_DIR}/summary_T6b_v3.csv",
    index=False
)

print("=== PART 4 COMPLETED — Evaluation Saved ===")
