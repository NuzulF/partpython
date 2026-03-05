############################################################
#                T6b_v3 — PART 3
#     USER SIMILARITY, ITEM SIMILARITY, PREDICTION
############################################################

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path

current_dir = os.getcwd()


# ==========================================================
# Load DATA NR & VECTORS (from PART 1 & PART 2)
# ==========================================================
OUT_DIR = os.path.join(current_dir,"T6b_v3_FULL")

df = pd.read_csv(f"{OUT_DIR}/data_new1_ldanan_NR_T6b_v3.csv", encoding="latin1")
train_df = pd.read_csv(f"{OUT_DIR}/train_df_T6b_v3.csv")
test_df  = pd.read_csv(f"{OUT_DIR}/test_df_T6b_v3.csv")

topic_profile_user = pd.read_csv(f"{OUT_DIR}/topic_profile_user_T6b_v3.csv", index_col=0)
item_vec_T6b       = pd.read_csv(f"{OUT_DIR}/item_vec_T6b_v3.csv", index_col=0)

print("Loaded all PART 3 inputs.")



############################################################
# 1. USER SIMILARITY — CF MEAN-CENTERED (from new_rating)
############################################################
def build_user_sim_from_newrating(df):
    pivot = df.pivot_table(
        index="id_reviewer",
        columns="id_dtw",
        values="new_rating",
        aggfunc="mean"
    )

    means = pivot.mean(axis=1)
    mat = pivot.sub(means, axis=0).fillna(0)

    sim = cosine_similarity(mat)
    sim_df = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)

    return sim_df, pivot, means


user_sim_CF, user_pivot, user_means = build_user_sim_from_newrating(train_df)



############################################################
# 2. USER SIMILARITY — LDA SEMANTIC (10D)
############################################################
topic_user_mat = topic_profile_user.values  # shape: (n_user, 10)

user_sim_LDA = pd.DataFrame(
    cosine_similarity(topic_user_mat),
    index=topic_profile_user.index,
    columns=topic_profile_user.index
)


############################################################
# 3. FINAL USER SIM (β-semantic blend)
############################################################
beta_sem = 0.0   # user semantic blend (can change easily)

user_sim_T6b = (1 - beta_sem) * user_sim_CF + beta_sem * user_sim_LDA

user_sim_T6b.to_csv(f"{OUT_DIR}/user_sim_T6b_v3.csv")



############################################################
# 4. ITEM SIMILARITY — 17D semantic vector
############################################################
item_sim_T6b = pd.DataFrame(
    cosine_similarity(item_vec_T6b.values),
    index=item_vec_T6b.index,
    columns=item_vec_T6b.index
)

item_sim_T6b.to_csv(f"{OUT_DIR}/item_sim_T6b_v3.csv")



############################################################
# 5. PREDICT UBCF (from new_rating)
############################################################
def predict_UBCF(u, i, pivot, sim_df, means, k=20):

    if u not in pivot.index or i not in pivot.columns:
        return np.nan

    col = pivot[i]
    rated_by = col[col.notna()].index
    rated_by = rated_by[rated_by != u]

    if len(rated_by) == 0:
        return np.nan

    sims = sim_df.loc[u, rated_by]
    nbr = sims.abs().sort_values(ascending=False).head(k).index

    sims_n = sims[nbr]
    r_n = pivot.loc[nbr, i]
    m_n = means.loc[nbr]

    num = np.sum(sims_n * (r_n - m_n))
    den = np.sum(np.abs(sims_n))

    if den == 0:
        return np.nan

    pred = means.loc[u] + num/den
    return pred



############################################################
# 6. PREDICT IBCF (from new_rating)
############################################################
def predict_IBCF(u, i, pivot, sim_matrix, k=20):

    if u not in pivot.index or i not in pivot.columns:
        return np.nan

    user_vec = pivot.loc[u]
    mask = user_vec.notna() & (user_vec.index != i)

    sims = sim_matrix.loc[i, user_vec.index[mask]]
    if sims.empty:
        return np.nan

    nbr = sims.abs().sort_values(ascending=False).head(k).index
    sims_n = sims[nbr]
    r_n = user_vec[nbr]

    num = np.sum(sims_n * r_n)
    den = np.sum(np.abs(sims_n))

    if den == 0:
        return np.nan

    pred = num/den
    return pred



############################################################
# 7. HYBRID PREDICTION (SEEN)
############################################################
theta_h = 0.5   # hybrid weight UBCF–IBCF

rows_seen = []

for idx, row in test_df.iterrows():
    u = row["id_reviewer"]
    i = row["id_dtw"]
    actual = row["rating"]

    pu = predict_UBCF(u, i, user_pivot, user_sim_T6b, user_means)
    pi = predict_IBCF(u, i, user_pivot, item_sim_T6b)

    if np.isnan(pu) and not np.isnan(pi):
        pu = pi
    if np.isnan(pi) and not np.isnan(pu):
        pi = pu

    if np.isnan(pu) and np.isnan(pi):
        pred = np.nan
    else:
        pred = theta_h * pu + (1 - theta_h) * pi

    # CLIP ke 0–5
    pred_clipped = float(np.clip(pred, 0, 5))

    rows_seen.append([u, i, actual, pred, pred_clipped])

pred_seen_T6b = pd.DataFrame(
    rows_seen,
    columns=["id_reviewer","id_dtw","Actual",
             "Pred_raw","Pred_clipped"]
)

pred_seen_T6b.to_csv(f"{OUT_DIR}/pred_seen_T6b_v3.csv", index=False)



############################################################
# 8. PREDICTION UNSEEN (TOP-N MATRIX)
############################################################
pivot_item = train_df.pivot_table(
    index="id_reviewer",
    columns="id_dtw",
    values="new_rating",
    aggfunc="mean"
).fillna(0)

pivot_item = pivot_item.reindex(columns=item_vec_T6b.index, fill_value=0)

unseen_matrix = pd.DataFrame(index=pivot_item.index,
                             columns=pivot_item.columns)

for u in pivot_item.index:
    rated = pivot_item.loc[u]
    rated = rated[rated > 0].index

    unrated = [it for it in pivot_item.columns if it not in rated]

    for it in unrated:
        val = predict_IBCF(u, it, pivot_item, item_sim_T6b)
        if np.isnan(val):
            unseen_matrix.loc[u, it] = np.nan
        else:
            unseen_matrix.loc[u, it] = np.clip(val, 0, 5)

unseen_matrix.to_csv(f"{OUT_DIR}/pred_unseen_T6b_v3.csv")

print("=== PART 3 COMPLETED — USER/ITEM SIM + PREDICTION READY ===")
