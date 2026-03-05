############################################################
#                T6b_v3 — PART 1
#    LOAD DATA + RECALCULATE NEW RATING (T1–T3 GOLDEN)
############################################################

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

current_dir = os.getcwd()

# ==========================================================
# Output folder
# ==========================================================
OUT_DIR = os.path.join(current_dir, "T6b_v3_FULL")
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# Load original dataset
# ==========================================================
DATA_PATH = Path.cwd() / "data" / "data_new1_ldanan.csv"
df = pd.read_csv(DATA_PATH, encoding="latin1")
print("Loaded df:", df.shape)


############################################################
# Helper: ensure column exists
############################################################
def ensure_col(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    return df


############################################################
# Golden NEW RATING — formula T1–T3 official
############################################################
def recalc_new_rating(df,
                      omega1, omega2, omega3,
                      beta_NR, alpha_NR):

    # Pastikan semua kolom tersedia
    df = ensure_col(df, "rating")
    df = ensure_col(df, "sentimen")
    df = ensure_col(df, "new_jarak")
    df = ensure_col(df, "lda_rev")
    df = ensure_col(df, "score_ibjoint")

    # (1) Base mix = rating + sentimen + jarak
    base_mix = (
        omega1 * df["rating"] +
        omega2 * df["sentimen"] +
        omega3 * df["new_jarak"]
    )

    # (2) User part = kombinasi base_mix + lda_rev
    user_part = (1 - beta_NR) * base_mix + beta_NR * df["lda_rev"]

    # (3) Final new_rating = user_part + ibjoint
    new_rating = (
        alpha_NR * user_part +
        (1 - alpha_NR) * df["score_ibjoint"]
    )
    return new_rating


############################################################
# Golden parameters (FINAL)
############################################################
omega1 = 0.1
omega2 = 0.2
omega3 = 0.7

beta_NR  = 0.0    # pure (rating+sentimen+jarak)
alpha_NR = 0.5    # 50% user_part + 50% ibjoint


############################################################
# Apply new rating
############################################################
df["new_rating"] = recalc_new_rating(
    df,
    omega1, omega2, omega3,
    beta_NR, alpha_NR
)

print("Range new_rating (raw):", df["new_rating"].min(), df["new_rating"].max())


############################################################
# Save NR dataset for next parts
############################################################
NR_PATH = os.path.join(OUT_DIR, "data_new1_ldanan_NR_T6b_v3.csv")
df.to_csv(NR_PATH, index=False)
print("Saved:", NR_PATH)