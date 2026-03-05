############################################################
#                T6b_v3 — PART 5
#           TOP-5 RECOMMENDATION GENERATION
############################################################

import numpy as np
import pandas as pd
import os
from pathlib import Path

current_dir = os.getcwd()

OUT_DIR = os.path.join(current_dir,"T6b_v3_FULL")

# ==========================================================
# Load unseen predictions (already clipped 0–5)
# ==========================================================
unseen_matrix = pd.read_csv(f"{OUT_DIR}/pred_unseen_T6b_v3.csv", index_col=0)

# Load raw dataset to retrieve user/destination names
df_raw = pd.read_csv("/content/drive/MyDrive/MODEL TMSA/TRIPJAVA/data_new1_ldanan.csv",
                     encoding="latin1")

lookup_user_name = df_raw.groupby("id_reviewer")["reviewer"].first()
lookup_dest_name = df_raw.groupby("id_dtw")["nama DTW"].first()

print("Loaded unseen_matrix and lookup tables.")



############################################################
#            PART 5A — TOP-5 SEDERHANA UNTUK SEMUA USER
############################################################
top5_rows = []

for u in unseen_matrix.index:

    scores = unseen_matrix.loc[u].dropna()
    if len(scores) == 0:
        continue

    # ambil top-5 tertinggi
    top5 = scores.sort_values(ascending=False).head(5)

    for item_id, pred in top5.items():
        top5_rows.append([u, item_id, pred])

df_top5_simple = pd.DataFrame(
    top5_rows,
    columns=["id_reviewer", "id_dtw", "Pred_T6b_v3"]
)

df_top5_simple.to_csv(
    f"{OUT_DIR}/Top5_simple_T6b_v3.csv",
    index=False
)

print("Top-5 SIMPLE T6b_v3 saved.")



############################################################
#            PART 5B — TOP-5 RICH (dengan nama + scale)
############################################################
top5_rich_rows = []

for u in unseen_matrix.index:

    scores = unseen_matrix.loc[u].dropna()
    if len(scores) == 0:
        continue

    top5 = scores.sort_values(ascending=False).head(5)

    for item_id, pred_norm in top5.items():

        # pred_norm sudah skala 0–5 → tidak perlu scaling tambahan
        pred_scaled = pred_norm  # tetap simpan kolom terpisah

        top5_rich_rows.append([
            u,
            lookup_user_name.get(u, "NA"),
            item_id,
            lookup_dest_name.get(item_id, "NA"),
            None,                # actual rating (unseen)
            pred_norm,
            pred_scaled
        ])

df_top5_rich = pd.DataFrame(
    top5_rich_rows,
    columns=[
        "id_reviewer",
        "nama_reviewer",
        "id_dtw",
        "nama_dtw",
        "Actual_Rating",
        "Pred_norm",
        "Pred_scaled_0_5"
    ]
)

df_top5_rich.to_csv(
    f"{OUT_DIR}/Top5_RICH_T6b_v3.csv",
    index=False
)

print("Top-5 RICH T6b_v3 saved.")



############################################################
#        PART 5C — TOP-5 RICH UNTUK USER TERTENTU
############################################################
# GANTI ID USER DI SINI
u = "r424"

if u in unseen_matrix.index:
    scores = unseen_matrix.loc[u].dropna()
    topk = scores.sort_values(ascending=False).head(5)

    rows_user = []
    for item_id, pred_norm in topk.items():
        rows_user.append([
            u,
            lookup_user_name.get(u, "NA"),
            item_id,
            lookup_dest_name.get(item_id, "NA"),
            None,
            pred_norm,
            pred_norm  # clipped & already 0–5
        ])

    df_top5_user = pd.DataFrame(
        rows_user,
        columns=[
            "id_reviewer",
            "nama_reviewer",
            "id_dtw",
            "nama_dtw",
            "Actual_Rating",
            "Pred_norm",
            "Pred_scaled_0_5"
        ]
    )

    df_top5_user.to_csv(
        f"{OUT_DIR}/Top5_user_{u}_T6b_v3.csv",
        index=False
    )

    print(f"Top-5 for user {u} saved.")
    display(df_top5_user)

else:
    print(f"User {u} not found in unseen_matrix.")
