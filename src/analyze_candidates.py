#!/usr/bin/env python3
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path):
    return pd.read_excel(path)

def preprocess(df):
    # Excel cols: P→15, Q→16, T→19
    archetype_col = df.columns[15]
    q_col         = df.columns[16]
    t_col         = df.columns[19]
    
    # combine open‑ended responses
    texts = (df[q_col].fillna('') + ' ' + df[t_col].fillna('')).tolist()
    return archetype_col, texts

def compute_similarity(texts):
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform(texts)
    return cosine_similarity(tfidf)

def split_by_archetype(df, archetype_col):
    idx = lambda label: df[df[archetype_col] == label].index.tolist()
    return {
        'LoP': idx('Life of the Party'),
        'St' : idx('Storyteller'),
        'QO' : idx('Quiet Observer'),
        'FC' : idx('Food Critic'),
    }

def build_single_table(df, archetype_col, sim, splits, lo):
    """
    Simulate one table anchored on Life‑of‑the‑Party index `lo`.
    Returns a list of 5 row‑indices: [lo, St, St (or FC), QO, QO (or FC)].
    """
    # copy buckets so we don’t destroy original splits
    local = {k: splits[k].copy() for k in splits}
    used = set([lo])
    members = [lo]

    # 2 storyteller slots (fallback to FC)
    for _ in range(2):
        cands = [i for i in local['St'] if i not in used]
        if not cands:
            cands = [i for i in local['FC'] if i not in used]
            bucket = 'FC'
        else:
            bucket = 'St'
        # pick best average similarity to current members
        scores = [np.mean([sim[i, m] for m in members]) for i in cands]
        pick   = cands[int(np.argmax(scores))]
        members.append(pick)
        used.add(pick)
        local[bucket].remove(pick)

    # 2 quiet observer slots (fallback to FC)
    for _ in range(2):
        cands = [i for i in local['QO'] if i not in used]
        if not cands:
            cands = [i for i in local['FC'] if i not in used]
            bucket = 'FC'
        else:
            bucket = 'QO'
        scores = [np.mean([sim[i, m] for m in members]) for i in cands]
        pick   = cands[int(np.argmax(scores))]
        members.append(pick)
        used.add(pick)
        local[bucket].remove(pick)

    return members

def score_table(members, sim):
    # sum of pairwise similarities (i<j)
    total = 0.0
    for ix, i in enumerate(members):
        for j in members[ix+1:]:
            total += sim[i, j]
    return total

if __name__ == '__main__':
    PATH = 'Sixth Seat Bid (Fey x Tlatolli) (Responses).xlsx'
    df = load_data(PATH)
    arch_col, texts = preprocess(df)
    sim = compute_similarity(texts)
    splits = split_by_archetype(df, arch_col)

    # 1) Find the BEST single table
    best_score = -1
    best_table = None
    for lo in splits['LoP']:
        tbl = build_single_table(df, arch_col, sim, splits, lo)
        sc  = score_table(tbl, sim)
        if sc > best_score:
            best_score = sc
            best_table = tbl

    # 2) Print primary table
    print(f"\nPrimary Table (score={best_score:.3f}):")
    for idx in best_table:
        name = df.loc[idx, df.columns[1]]  # “first & last name” is cols[1]
        arch = df.loc[idx, arch_col]
        print(f"  • {name}  ({arch})")

    # 3) Generate N backups by removing that LoP and repeating
    NUM_BACKUPS = 2
    backups = []
    # make a fresh copy of splits and drop the chosen LoP
    rem_splits = {k: splits[k].copy() for k in splits}
    chosen_lop = best_table[0]
    rem_splits['LoP'].remove(chosen_lop)

    for b in range(NUM_BACKUPS):
        if not rem_splits['LoP']:
            break
        # rerun best‑LoP selection on the reduced pool
        b_score = -1
        b_table = None
        for lo in rem_splits['LoP']:
            tbl = build_single_table(df, arch_col, sim, rem_splits, lo)
            sc  = score_table(tbl, sim)
            if sc > b_score:
                b_score = sc
                b_table = tbl
        backups.append((b_table, b_score))
        # remove that LoP to avoid reuse
        rem_splits['LoP'].remove(b_table[0])

    # 4) Print backups
    for i, (tbl, sc) in enumerate(backups, start=1):
        print(f"\nBackup #{i} Table (score={sc:.3f}):")
        for idx in tbl:
            name = df.loc[idx, df.columns[1]]
            arch = df.loc[idx, arch_col]
            print(f"  • {name}  ({arch})")
