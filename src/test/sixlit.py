import streamlit as st
import pandas as pd
import numpy as np
import math
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

from test_analyze_candidates import (
    load_form_submissions,
    preprocess_submissions,
    compute_similarity,
    split_by_archetype,
    build_single_table
)

st.title("Six Seat Table Builder")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = load_form_submissions(uploaded_file)
    st.write("Data", df)

    archetype, survey_responses_text = preprocess_submissions(df)
    sim = compute_similarity(survey_responses_text)
    splits = split_by_archetype(df, 'Archetype')

    # Step 1: Pick LoP
    if splits['LoP']:
        lotp_options = [(idx, df.loc[idx]['Name']) for idx in splits['LoP']]
        lotp_idx = st.selectbox(
            "Choose Life of the Party",
            options=lotp_options,
            format_func=lambda x: x[1]
        )
        lotp = lotp_idx[0]
        st.write("Building table starting with:", df.loc[lotp]['Name'])

        # Step 2: Show compatible candidates for each role and let user refine
        # Example for Storyteller
        compatible_storytellers = sorted(
            [(idx, sim[lotp][idx]) for idx in splits['St'] if idx != lotp],
            key=lambda x: x[1], reverse=True
        )
        st_options = [(idx, df.loc[idx]['Name']) for idx, score in compatible_storytellers[:5]]
        selected_st_idx = st.selectbox(
            "Choose Storyteller",
            options=st_options,
            format_func=lambda x: x[1]
        )
        selected_st = selected_st_idx[0]

        # Example for Food Critic
        compatible_fc = sorted(
            [(idx, sim[lotp][idx]) for idx in splits['FC'] if idx != lotp],
            key=lambda x: x[1], reverse=True
        )
        fc_options = [(idx, df.loc[idx]['Name']) for idx, score in compatible_fc[:5]]
        selected_fc_idx = st.selectbox(
            "Choose Food Critic",
            options=fc_options,
            format_func=lambda x: x[1]
        )
        selected_fc = selected_fc_idx[0]

        # You can repeat for other archetypes as needed

        # Step 3: Build table with user choices
        # You need to update build_single_table to accept selected_st and selected_fc
        selected_indices = build_single_table(
            df, 'Archetype', sim, splits, lotp,
            selected_st=selected_st,
            selected_fc=selected_fc
        )

        st.subheader("Dinner Table Arrangement")
        num = len(selected_indices)
        radius = 180
        center_x, center_y = 250, 250

        cards_html = """
        <style>
        .table-circle {
            position: relative;
            width: 500px;
            height: 500px;
            margin: 40px auto;
            background: #f5f5f5;
            border-radius: 50%;
            box-shadow: 0 2px 12px #0001;
        }
        .candidate-card-circle {
            position: absolute;
            width: 140px;
            height: 140px;
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 2px 12px #0001;
            text-align: center;
            font-family: 'Times New Roman', Times, serif;
            letter-spacing: -0.074em;
            border: 1.5px solid #1612d3;
            padding: 12px 4px;
            transform: translate(-50%, -50%);
            overflow: hidden;
            z-index: 2;
            transition: transform 0.2s ease;
        }
        .candidate-card-circle:hover {
            transform: translate(-50%, -50%) scale(1.05);
            box-shadow: 0 4px 16px #0002;
            cursor: pointer;
        }
        </style>
        <div class="table-circle">
        """

        # Central anchor
        cards_html += """
        <div style="
            position: absolute;
            left: 250px;
            top: 250px;
            transform: translate(-50%, -50%);
            background: #1612d3;
            color: #FFEED6;
            padding: 12px 20px;
            border-radius: 50%;
            font-weight: bold;
            box-shadow: 0 2px 12px #0002;
            z-index: 1;
        ">
            Table
        </div>
        """

        for i, idx in enumerate(selected_indices):
            angle = 2 * math.pi * i / num
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            name = html.escape(df.loc[idx]['Name'])
            archetype = html.escape(df.loc[idx]['Archetype'])
            avatar_url = df.loc[idx]['avatar'] if 'avatar' in df.columns and pd.notnull(df.loc[idx]['avatar']) else "https://via.placeholder.com/60"
            score = int(sim[lotp][idx] * 100)

            cards_html += f"""
            <div class="candidate-card-circle" style="left:{x}px; top:{y}px;">
                <h3 style="margin-bottom: 0.5rem;">{name}</h3>
                <div style="font-size: 1rem; margin-bottom: 0.5rem;">{archetype}</div>
                <img src="{avatar_url}" width="60" style="border-radius:50%;margin-bottom:0.5rem;"/>
                <div style="
                    position: absolute;
                    bottom: 6px;
                    right: 6px;
                    background: #1612d3;
                    color: #FFEED6;
                    font-size: 0.75rem;
                    padding: 4px 8px;
                    border-radius: 12px;
                ">
                    {score}%
                </div>
            </div>
            """

        cards_html += "</div>"
        components.html(cards_html, height=600)