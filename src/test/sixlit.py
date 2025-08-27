import streamlit as st
import pandas as pd
import numpy as np
import math
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
        background-color: #fff !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #1612d3;
        font-family: 'Times New Roman', Times, serif;
        letter-spacing: -0.074em;
    }
    .stButton>button {
        background-color: #1612d3;
        color: #FFEED6;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        font-family: 'Times New Roman', Times, serif;
        letter-spacing: -0.074em;
    }
    /* Card styling */
    .candidate-card {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 12px #0001;
        padding: 28px 16px;
        text-align: center;
        margin: 12px;
        min-width: 180px;
        width: 100%;
        max-width: 260px;
        display: inline-block;
        font-family: 'Times New Roman', Times, serif;
        letter-spacing: -0.074em;
        border: 2px solid #1612d3;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Six Seat Table Builder")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = load_form_submissions(uploaded_file)
    st.write("Data", df)
    archetype, survey_responses_text = preprocess_submissions(df)
    sim = compute_similarity(survey_responses_text)
    splits = split_by_archetype(df, 'Archetype')
    st.write("Archetype Splits:", splits)
    if splits['LoP']:
        lotp_options = [(idx, df.loc[idx]['Name']) for idx in splits['LoP']]
        lotp_idx = st.selectbox(
            "Choose Life of the Party",
            options=lotp_options,
            format_func=lambda x: x[1]
        )
        lotp = lotp_idx[0]
        st.write("Building table starting with:", df.loc[lotp]['Name'])
        selected_indices = build_single_table(df, 'Archetype', sim, splits, lotp)
        st.subheader("Selected Table Members")
        cols = st.columns(len(selected_indices))
        for col, idx in zip(cols, selected_indices):
            with col:
                st.markdown(
                    f"""
                    <div class="candidate-card">
                        <h3 style="margin-bottom: 0.5rem; color: #1612d3;">{df.loc[idx]['Name']}</h3>
                        <div style="color: #1612d3; font-size: 1rem; margin-bottom: 0.5rem;">{df.loc[idx]['Archetype']}</div>
                        {"<img src='" + df.loc[idx]['avatar'] + "' width='80' style='border-radius:50%;margin-bottom:0.5rem;'/>" if 'avatar' in df.columns and pd.notnull(df.loc[idx]['avatar']) else ""}
                    </div>
                    """,
                    unsafe_allow_html=True
                )