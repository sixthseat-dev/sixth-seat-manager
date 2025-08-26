import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_form_submissions(file) -> pd.DataFrame:
    """
    Load and validate form submissions from csv.
    Accepts a file path or a file-like object (e.g., Streamlit UploadedFile).
    """
    if hasattr(file, "read"):  # file-like object (Streamlit)
        return pd.read_csv(file)
    else:  # file path as string or Path
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        return pd.read_csv(path)

def preprocess_submissions(df):

    archetype = df['Archetype']
    # print(archetype)
    personality_response_talktopic = df['TalkTopic']
    # print(personality_response_talktopic)
    personality_response_tabletalk = df['TableTalk']
    survey_responses_text = (personality_response_talktopic + " " + personality_response_tabletalk)
    # print(survey_responses_text.tolist())
    return archetype, survey_responses_text.tolist()

def compute_similarity(texts):
    """do some stuff"""
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform(texts)
    return cosine_similarity(tfidf)

def get_indices_by_label(df, archetype_col, label):
    """do something"""
    # print(df[archetype_col]), label
    return df[df[archetype_col] == label].index.tolist()

def split_by_archetype(df, archetype_col):
    return {
        'LoP': get_indices_by_label(df, archetype_col, 'Life of the Party'),
        'St':  get_indices_by_label(df, archetype_col, 'Storyteller'),
        'QO':  get_indices_by_label(df, archetype_col, 'Quiet Obsever'),
        'FC':  get_indices_by_label(df, archetype_col, 'Food Critic'),
    }

def pick_best_candidate(candidates, members, sim):
    "Return the candidate with the highest average similarity score to the current members"
    if not candidates:
        return None
    # print("Candidates:", candidates)
    # print("Members:", members)
    # print("Similarity matrix: \n", sim)

    scores = [np.mean([sim[i, m] for m in members]) for i in candidates]
    # print("Scores: " + str(scores))
    return candidates[int(np.argmax(scores))]

def build_single_table(df, archetype_col, sim, splits, lotp):
    # print("Splits", splits)
    for k in splits:
        local = {k: splits[k].copy() for k in splits}
    # print("Local", local)
    used = set([lotp])
    members = [lotp]
    
    for _ in range(2):
        cands = [i for i in local['St'] if i not in used]
        #print("Cands: ", cands)
        if not cands:
            cands = [i for i in local['FC'] if i not in used]
            bucket = 'FC'
        else:
            bucket = 'St'
        scores = [np.mean([sim[i, m] for m in members]) for i in cands ]
        pick_index = int(np.argmax(scores))
        #pick = cands[int(np.argmax(scores))]
        pick = cands[pick_index]
        pick_score = scores[pick_index]
        print(f"Pick: {pick}, Similarity Score: {pick_score*100:.1f}%")
        members.append(pick)
        used.add(pick)
        local[bucket].remove(pick)
        print(members)

    for _ in range(2):
        cands = [i for i in local['QO'] if i not in used]
        if not cands:
            cands = [i for i in local['FC'] if i not in used]
            bucket = 'FC'
        else:
            bucket = 'QO'
        scores = [np.mean([sim[i, m] for m in members]) for i in cands]
        pick_index = int(np.argmax(scores))
        pick = cands[pick_index]
        members.append(pick)
        used.add(pick)
        local[bucket].remove(pick)
        print(members)
    return members


if __name__ == '__main__':
    PATH = '../../data/mock_sixth_seat_bid_responses.csv'
    df = load_form_submissions(PATH)
    # print(df.head(5))
    archetype, survey_responses_text = preprocess_submissions(df)
    # print(archetype)
    sim = compute_similarity(survey_responses_text)
    # print(sim)
    #print(get_indices_by_label(df, 'Archetype', 'Life of the Party'))
    splits = split_by_archetype(df, 'Archetype' )
    print("Splits: ", splits)
    candidates = splits['LoP'][:3]
    print("Candidates: " + str(candidates))
    members = [candidates[0]]
    print("Members: " + str(members))
    picked_index = (pick_best_candidate(candidates, members, sim))
    print("Picked index:", picked_index)
    print("Picked candidate data:\n", df.loc[picked_index])
    lotp = splits['LoP'][0]
    # print("Lop: ", lotp)
    build_single_table(df, 'Archetype', sim, splits, lotp )

