import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_DIR, MODEL_DIR


############################################################
# HELPER
############################################################

def ensure_col(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    return df


############################################################
# NEW RATING
############################################################

def recalc_new_rating(df,
                      omega1=0.1,
                      omega2=0.2,
                      omega3=0.7,
                      beta_NR=0.0,
                      alpha_NR=0.5):

    df = ensure_col(df, "rating")
    df = ensure_col(df, "sentimen")
    df = ensure_col(df, "new_jarak")
    df = ensure_col(df, "lda_rev")
    df = ensure_col(df, "score_ibjoint")

    base_mix = (
        omega1 * df["rating"] +
        omega2 * df["sentimen"] +
        omega3 * df["new_jarak"]
    )

    user_part = (1 - beta_NR) * base_mix + beta_NR * df["lda_rev"]

    new_rating = (
        alpha_NR * user_part +
        (1 - alpha_NR) * df["score_ibjoint"]
    )

    return new_rating


## SPLIT

def per_user_split(df, user_col='id_reviewer', test_size=0.2, seed=7):

    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []

    for u, idx in df.groupby(user_col).indices.items():

        idx = np.array(list(idx))

        if len(idx) <= 2:
            train_idx.extend(idx)
            continue

        n_test = max(1, int(test_size * len(idx)))

        test = rng.choice(idx, n_test, replace=False)

        train = np.setdiff1d(idx, test)

        train_idx.extend(train)
        test_idx.extend(test)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    return train_df, test_df


############################################################
# TRAIN MODEL
############################################################

def train_model():

    df = pd.read_csv(
        os.path.join(DATA_DIR, "data_new1_ldanan.csv"),
        encoding="latin1"
    )

    df["new_rating"] = recalc_new_rating(df)

    pivot = df.pivot_table(
        index="id_reviewer",
        columns="id_dtw",
        values="new_rating",
        aggfunc="mean"
    )

    means = pivot.mean(axis=1)

    mat = pivot.sub(means, axis=0).fillna(0)

    sim = cosine_similarity(mat)

    user_sim = pd.DataFrame(
        sim,
        index=pivot.index,
        columns=pivot.index
    )

    with open(os.path.join(MODEL_DIR, "pivot.pkl"), "wb") as f:
        pickle.dump(pivot, f)

    with open(os.path.join(MODEL_DIR, "user_means.pkl"), "wb") as f:
        pickle.dump(means, f)

    with open(os.path.join(MODEL_DIR, "user_sim.pkl"), "wb") as f:
        pickle.dump(user_sim, f)

    print("Training completed")