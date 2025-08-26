import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from test_analyze_candidates import (
    load_form_submissions,
    preprocess_submissions,
    compute_similarity,
    split_by_archetype,
    pick_best_candidate,
    build_single_table
)

st.markdown("""
    <style>
    .stApp {
        background-color: #fdf6ee !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #222;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #2a6fdb;
        color: #222;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Paste your functions here: load_form_submissions, preprocess_submissions, etc.

st.title("Six Seat Table Builder")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = load_form_submissions(uploaded_file)
    st.write("Data", df)
    archetype, survey_responses_text = preprocess_submissions(df)
    sim = compute_similarity(survey_responses_text)
    splits = split_by_archetype(df, 'Archetype')
    st.write(" Archetype Splits: ", splits)
    lotp = splits['LoP'][0] if splits['LoP'] else None
    if lotp is not None:
        st.write("Building table starting with:", df.loc[lotp]['Name'])
        selected_indices = build_single_table(df, 'Archetype', sim, splits, lotp)
        st.subheader("Selected Table Members")
        for idx in selected_indices:
            st.write(df.loc[idx])