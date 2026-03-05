############################################################
#                T6b_v3 — PART 2
#    USER SPLIT + LDA LOAD + VECTOR USER/ITEM (17D item)
############################################################

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import pickle
from pathlib import Path

current_dir = os.getcwd()


# ==========================================================
# Load NR Dataset from PART 1
# ==========================================================
DATA_NR = Path.cwd() / "T6b_v3_FULL" / "data_new1_ldanan_NR_T6b_v3.csv"
df = pd.read_csv(DATA_NR, encoding="latin1")
print("Loaded NR df:", df.shape)


# ==========================================================
# Output folder
# ==========================================================
OUT_DIR = os.path.join(current_dir,"T6b_v3_FULL")
os.makedirs(OUT_DIR, exist_ok=True)


############################################################
#                1. USER SPLIT (SAME AS T6a)
############################################################
def per_user_split(df, user_col='id_reviewer', test_size=0.2, min_items=2, seed=7):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for u, idxs in df.groupby(user_col).indices.items():
        idxs = np.array(list(idxs))
        if len(idxs) <= min_items:
            train_idx.extend(idxs)
            continue

        n_test = max(1, int(round(test_size * len(idxs))))
        test_sel = rng.choice(idxs, size=n_test, replace=False)
        train_sel = np.setdiff1d(idxs, test_sel)

        train_idx.extend(train_sel)
        test_idx.extend(test_sel)

    return np.array(train_idx), np.array(test_idx)


train_idx, test_idx = per_user_split(df, seed=7)
train_df = df.iloc[train_idx].copy()
test_df  = df.iloc[test_idx].copy()

# Save split
train_df.to_csv(os.path.join(OUT_DIR, "train_df_T6b_v3.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test_df_T6b_v3.csv"), index=False)


############################################################
#        2. LOAD LDA MODELS (review 10D, deskripsi 7D)
############################################################
BASE = os.path.join(current_dir,"data")

lda_model_rev  = LdaModel.load(os.path.join(BASE, "lda_model_review.gensim"))   # 10 topics
lda_model_desc = LdaModel.load(os.path.join(BASE, "lda_model.gensim"))         # 7 topics

dictionary_rev  = corpora.Dictionary.load(os.path.join(BASE, "dictionary_review.gensim"))
dictionary_desc = corpora.Dictionary.load(os.path.join(BASE, "dictionary_des.gensim"))

with open(os.path.join(BASE, "stopwords_des.pkl"), "rb") as f:
    stop_words = pickle.load(f)


def preprocess_text(txt):
    if not isinstance(txt, str):
        txt = str(txt)
    return [w for w in simple_preprocess(txt) if w not in stop_words]


############################################################
#    3. TOPIC PROFILE USER (10D from LDA-review)
############################################################
def infer_topic_profile_user(df):
    vecs = []
    for uid, g in df.groupby("id_reviewer"):
        tokens = []
        for tx in g["review"]:
            tokens.extend(preprocess_text(tx))

        bow = dictionary_rev.doc2bow(tokens)
        dist = lda_model_rev.get_document_topics(bow, minimum_probability=0)

        v = np.zeros(lda_model_rev.num_topics)  # 10D
        for t, p in dist:
            v[t] = p

        vecs.append((uid, v))

    prof = {u: v for u, v in vecs}
    dfp = pd.DataFrame.from_dict(prof, orient="index")
    dfp.columns = [f"topic_rev_{i+1}" for i in range(dfp.shape[1])]
    return dfp


topic_profile_user = infer_topic_profile_user(train_df)
topic_profile_user.to_csv(os.path.join(OUT_DIR, "topic_profile_user_T6b_v3.csv"))


############################################################
#    4. TOPIC PROFILE ITEM (7D desc + 10D review → 17D)
############################################################
def infer_topic_profile_item_desc(df):
    vecs = []
    for it, g in df.groupby("id_dtw"):
        tokens = []
        for tx in g["deskripsi"]:
            tokens.extend(preprocess_text(tx))

        bow = dictionary_desc.doc2bow(tokens)
        dist = lda_model_desc.get_document_topics(bow, minimum_probability=0)

        v = np.zeros(lda_model_desc.num_topics)  # 7D
        for t, p in dist:
            v[t] = p

        vecs.append((it, v))

    prof = {i: v for i, v in vecs}
    dfp = pd.DataFrame.from_dict(prof, orient="index")
    dfp.columns = [f"topic_desc_{i+1}" for i in range(dfp.shape[1])]
    return dfp


topic_profile_item_desc = infer_topic_profile_item_desc(train_df)
topic_profile_item_desc.to_csv(os.path.join(OUT_DIR, "topic_profile_item_desc_T6b_v3.csv"))


# --- ITEM REVIEW 10D ---
def infer_topic_profile_item_rev(df):
    vecs = []
    for it, g in df.groupby("id_dtw"):
        tokens = []
        for tx in g["review"]:
            tokens.extend(preprocess_text(tx))

        bow = dictionary_rev.doc2bow(tokens)
        dist = lda_model_rev.get_document_topics(bow, minimum_probability=0)

        v = np.zeros(lda_model_rev.num_topics)  # 10D
        for t,p in dist:
            v[t] = p

        vecs.append((it, v))

    prof = {i: v for i, v in vecs}
    dfp = pd.DataFrame.from_dict(prof, orient="index")
    dfp.columns = [f"topic_revitem_{i+1}" for i in range(dfp.shape[1])]
    return dfp


topic_profile_item_rev = infer_topic_profile_item_rev(train_df)
topic_profile_item_rev.to_csv(os.path.join(OUT_DIR, "topic_profile_item_rev_T6b_v3.csv"))


############################################################
#      5. BUILD ITEM VECTOR T6b (17D)
#          v_item_T6b(i) = [desc_7D, rev_10D]
############################################################
item_vec_T6b = pd.concat(
    [
        topic_profile_item_desc,   # 7D
        topic_profile_item_rev     # 10D
    ],
    axis=1
).fillna(0)

item_vec_T6b.to_csv(os.path.join(OUT_DIR, "item_vec_T6b_v3.csv"))


print("=== PART 2 COMPLETED — USER 10D & ITEM 17D READY ===")