from pt1 import *
from pt2 import *
from pt3 import *
from pt4 import *
from pt5 import *

############################################################
#        PART 5C — TOP-5 RICH UNTUK USER TERTENTU
############################################################
# GANTI ID USER DI SINI
u = "r425"

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
        f"{OUT_DIR}/Top5_user_{u}_T6b_v3_id425.csv",
        index=False
    )

    print(f"Top-5 for user {u} saved.")
    display(df_top5_user)

else:
    print(f"User {u} not found in unseen_matrix.")