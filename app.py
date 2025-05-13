# -*- coding: utf-8 -*-
"""
Created on Thu May  1 01:36:28 2025
@author: AWC Labs
"""

import re
import streamlit as st
from sklearn.linear_model import LogisticRegression, SGDClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import uuid
import requests

# Generate a unique session ID for the user
session_id = str(uuid.uuid4())

# Google Form POST URL
form_url = "https://docs.google.com/forms/u/0/d/1LbiPqyGcf5dSjF7CntQQNN9HloumgcncQ-Rj2xCNFrk/formResponse"

# Integration of Google Form: Replace with  entry.xxxxx IDs from Google Form
ENTRY_SESSION = "entry.153544605"
ENTRY_LOG = "entry.588008680"
ENTRY_FEEDBACK = "entry.2051031519"

# Page Setup
st.set_page_config(page_title="🤖CaptAI", layout="centered")
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>🎭 CaptAI\u2122</h1>
        <p style='font-size: 16px; color: gray;'>
            Powered by Dual Machine Learning Models for Nuanced, Fast, and Reliable Sentiment Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

from utils.utils import clean_text
import joblib

# Load models and vectorizer
almirax_model = joblib.load('Alx_model.pkl')
alekxia = joblib.load('Alx_model.pkl')
vectorizer = joblib.load('text_vectorizer.pkl')

models = {
    'Almirax': almirax_model,
    'Alekxia': alekxia
}


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("CaptAI\u2122 is powered by two core models: **Almirax** and **Alekxia**.")
    model_choice = st.selectbox("Chose a Model", list(models.keys()))
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
    st.caption("Adjust model and confidence threshold.")
    
    st.divider()

    # Model Info
    st.subheader("ℹ️ Model Details")

    if model_choice == 'Almirax':
        st.markdown("""
        - **Selected Model:** Almirax🧱
        - Almirax delivers clear, balanced, and trustworthy sentiment analysis.
        Built on proven logic, she offers interpretable insights for feedback, reviews, 
        and conversations. Almirax is ideal for users who value transparency and control, 
        as she brings calm precision to understanding language in regulated, high-trust environments
        """)
        
    elif model_choice == 'Alekxia':
        st.markdown("""
        - **Selected Model:** Alekxia⚡
        - Alekxia delivers fast, adaptive sentiment analysis at scale. 
        Designed for real-time environments like social media and live chat, she captures emotional shifts
        and trends instantly. Powered by a rapid-learning engine, 
        Alekxia is perfect for users who need quick, responsive insights without sacrificing context or nuance
        """)


# Initialize session state to track analysis status
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# Input Section
st.subheader("📝 Analyze Sentiment from  Text Review")
user_input = st.text_area(
    "📌 Please enter a review (e.g., movie opinion, product feedback):",
    placeholder="Type or paste text review here...",
    height=150
)

if user_input.strip().isnumeric():
    st.warning("⚠️ Please enter a valid text review. Pure numbers won't work.")

elif st.button(f"🧠 Analyze Sentiment with {model_choice}"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            cleaned_input = clean_text(user_input)
            model = models[model_choice]
            input_vector = vectorizer.transform([cleaned_input])

            proba = model.predict_proba(input_vector)[0] if hasattr(model, "predict_proba") else None
            if proba is not None:
                positive_proba = proba[1]
                prediction = 1 if positive_proba >= threshold else 0
                confidence = np.max(proba)
            else:
                prediction = model.predict(input_vector)[0]
                confidence = 1.0

        st.subheader("📈📉 Result")
        if prediction == 1:
            st.success("This review is Positive! 🎉", icon="👍")
            review_type = "**Positive**"
        else:
            st.error("This review is Negative. 😔", icon="👎")
            review_type = "**Negative**"

        st.metric(f"Hi, I am {model_choice}, my confidence about this {review_type} Review is:", f"{confidence*100:.1f}%")

        # Mark analysis as done and log details
        st.session_state.analysis_done = True
        st.session_state.user_log = f"Used model {model_choice}, prediction: {review_type}, confidence: {confidence:.2f}, input: {user_input[:100]}"

        # Feature Impact
        feature_names = vectorizer.get_feature_names_out()
        input_array = input_vector.toarray()[0]
        coef = model.coef_[0]
        word_contributions = input_array * coef
        top_indices = np.argsort(np.abs(word_contributions))[-5:][::-1]

        words_data = []
        for idx in top_indices:
            if input_array[idx] > 0:
                word = feature_names[idx]
                score = word_contributions[idx]
                impact = "✅Positive" if score > 0 else "❌Negative"
                words_data.append({
                    "Word": word,
                    "Impact": impact,
                    "Score": round(score, 4)
                })

        if words_data:
            st.markdown("Here are the key words in text review that shaped the sentiment result:")
            st.table(pd.DataFrame(words_data))
        else:
            st.info("No strong influential words found in the input.")

      
        st.divider()

        # Word Cloud
        st.subheader("🎨 Key Word Map")
        st.markdown(f"Here is the visual representation of the most prominent words in the text I analyzed to be **{review_type}**:")
        wc = WordCloud(width=400, height=200, background_color='white').generate(user_input)
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("👩‍🔬I hope you find all of these information to be helpful ")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# --- Collect Feedback only if analysis was done ---
if st.session_state.analysis_done:
    st.subheader("User Feedback✍️")
    user_feedback = st.text_area("User feedbacks are appreciated, please share  any comments about my usefulness, performance, or suggestions for improvement", height=100)
    if st.button(f"📩 Submit Feedback: {model_choice}"):
        form_data = {
            ENTRY_SESSION: session_id,
            ENTRY_LOG: st.session_state.user_log,
            ENTRY_FEEDBACK: user_feedback
        }

        response = requests.post(form_url, data=form_data)
        if response.status_code == 200:
            st.success("Feedback submitted! ✅")
        else:
            st.warning("Failed to submit feedback. 🚫")

# Footer
st.markdown("---")
st.markdown("""
   <div style="text-align: center; font-size: 0.85em; color: gray; line-height: 1.6em;">
    <strong>CaptAI™</strong>: Designed and Developed at <strong>AWC Labs</strong><br>
    📂 GitHub: <a href="https://github.com/Abdul-WriteCodes" target="_blank">AWC Labs</a><br>
    We appreciate voluntary support for this project via 
    <a href="https://www.buymeacoffee.com/abdul_writecodes" target="_blank" style="color: #ff5f1f; font-weight: bold;">☕BuyMeACoffee</a> 
    or 
    <a href="https://www.selar.com/showlove/awc-labs" target="_blank" style="color: #ff5f1f; font-weight: bold;">💖Selar</a><br>
    <strong>Disclaimer:</strong> Our platform does not collect or store personal data and information. 
    The feedback that is voluntarily given by users and collected by us is only used to improve the system.<br>
    © 2025 AWC Labs. All rights reserved.
</div>


""", unsafe_allow_html=True)
