import pickle
import os
import numpy as np
import pandas as pd

from config import MODEL_DIR, DATA_DIR


with open(os.path.join(MODEL_DIR,"pivot.pkl"),"rb") as f:
    pivot = pickle.load(f)

with open(os.path.join(MODEL_DIR,"user_means.pkl"),"rb") as f:
    means = pickle.load(f)

with open(os.path.join(MODEL_DIR,"user_sim.pkl"),"rb") as f:
    sim_df = pickle.load(f)


df_raw = pd.read_csv(
    os.path.join(DATA_DIR,"data_new1_ldanan.csv"),
    encoding="latin1"
)

lookup_dest_name = df_raw.groupby("id_dtw")["nama DTW"].first()


############################################################
# Predict single item
############################################################

def predict(user,item,k=20):

    if user not in pivot.index or item not in pivot.columns:
        return None

    col = pivot[item]

    rated_by = col[col.notna()].index
    rated_by = rated_by[rated_by != user]

    sims = sim_df.loc[user,rated_by]

    nbr = sims.abs().sort_values(ascending=False).head(k).index

    sims_n = sims[nbr]
    r_n = pivot.loc[nbr,item]
    m_n = means.loc[nbr]

    num = np.sum(sims_n * (r_n - m_n))
    den = np.sum(np.abs(sims_n))

    if den==0:
        return None

    pred = means.loc[user] + num/den

    return float(np.clip(pred,0,5))


############################################################
# Top N recommendation
############################################################

def recommend(user_id,top_n=5):

    if user_id not in pivot.index:
        return []

    user_scores = pivot.loc[user_id]

    unrated = user_scores[user_scores.isna()].index

    results = []

    for item in unrated:

        score = predict(user_id,item)

        if score is not None:

            results.append({
                "id_dtw":item,
                "nama DTW":lookup_dest_name.get(item,"Unknown"),
                "Pred_score":score,
                "Weighted_Average":score
            })

    results = sorted(
        results,
        key=lambda x:x["Weighted_Average"],
        reverse=True
    )

    return results[:top_n]