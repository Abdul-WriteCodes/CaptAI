# -*- coding: utf-8 -*-
"""
Created on Thu May  1 01:36:28 2025
@author: AWC Labs
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import uuid
import requests

# Caching session ID
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id = get_session_id()

# Cache model loading
@st.cache_resource
def load_models():
    return {
        'Almirax': joblib.load('almirax_pipeline.pkl'),
        'Alekxia': joblib.load('alekxia_pipeline.pkl')
    }

models = load_models()

# Google Form POST URL
form_url = "https://docs.google.com/forms/u/0/d/1LbiPqyGcf5dSjF7CntQQNN9HloumgcncQ-Rj2xCNFrk/formResponse"
ENTRY_SESSION = "entry.153544605"
ENTRY_LOG = "entry.588008680"
ENTRY_FEEDBACK = "entry.2051031519"

# Page Setup
st.set_page_config(page_title="🤖CaptAI", layout="centered")
st.markdown("""
    <div style='text-align: center;'>
        <h1>🎭 CaptAI™</h1>
        <p style='font-size: 16px; color: gray;'>
            Powered by Dual Machine Learning Models for Nuanced, Fast, and Reliable Sentiment Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("CaptAI™ is powered by two core models: **Almirax** and **Alekxia**.")
    model_choice = st.selectbox("Chose a Model", list(models.keys()))
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
    st.caption("Adjust model and confidence threshold.")
    st.divider()
    st.subheader("ℹ️ Model Details")
    if model_choice == 'Almirax':
        st.markdown("""
        - **Selected Model:** Almirax🧱  
        - Almirax delivers clear, balanced, and trustworthy sentiment analysis.
        Built on proven logic, she offers interpretable insights for feedback, reviews, 
        and conversations. Almirax is ideal for users who value transparency and control, 
        as she brings calm precision to understanding language in regulated, high-trust environments.
        """)
    elif model_choice == 'Alekxia':
        st.markdown("""
        - **Selected Model:** Alekxia⚡  
        - Alekxia delivers fast, adaptive sentiment analysis at scale. 
        Designed for real-time environments like social media and live chat, she captures emotional shifts
        and trends instantly. Powered by a rapid-learning engine, 
        Alekxia is perfect for users who need quick, responsive insights without sacrificing context or nuance.
        """)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

st.subheader("📝 Analyze Sentiment from Text Review")
user_input = st.text_area(
    "📌 Please enter a review (e.g., movie opinion, product feedback):",
    placeholder="Type or paste text review here...",
    height=150
)

analyze_button = st.button(f"🧠 Analyze Sentiment with {model_choice}")
if user_input.strip().isnumeric():
    st.warning("⚠️ Please enter a valid text review. Pure numbers won't work.")

if analyze_button and user_input.strip():
    with st.spinner("Analyzing..."):
        model = models[model_choice]
        try:
            proba = model.predict_proba([user_input])[0]
            positive_proba = proba[1]
            prediction = int(positive_proba >= threshold)
            confidence = positive_proba if prediction == 1 else 1 - positive_proba
        except AttributeError:
            prediction = model.predict([user_input])[0]
            confidence = 1.0

    st.subheader("📈📉 Result")
    review_type = "**Positive**" if prediction == 1 else "**Negative**"
    if prediction == 1:
        st.success("This review is Positive! 🎉", icon="👍")
    else:
        st.error("This review is Negative. 😔", icon="👎")

    st.metric(f"Hi, I am {model_choice}, my confidence about this {review_type} Review is:", f"{confidence*100:.1f}%")

    st.session_state.analysis_done = True
    st.session_state.user_log = f"Used model {model_choice}, prediction: {review_type}, confidence: {confidence:.2f}, input: {user_input[:100]}"

    try:
        vectorizer = model.named_steps['vectorizer']
        feature_names = vectorizer.get_feature_names_out()
        input_vector = vectorizer.transform([user_input])
        coef = model.named_steps['classifier'].coef_[0]
        input_array = input_vector.toarray()[0]
        word_contributions = input_array * coef
        top_indices = np.argsort(np.abs(word_contributions))[-5:][::-1]

        words_data = []
        for idx in top_indices:
            if input_array[idx] > 0:
                word = feature_names[idx]
                score = word_contributions[idx]
                impact = "✅Positive" if score > 0 else "❌Negative"
                words_data.append({"Word": word, "Impact": impact, "Score": round(score, 4)})

        if words_data:
            st.markdown("Key words that influenced sentiment:")
            st.table(pd.DataFrame(words_data))
        else:
            st.info("No strong influential words found in the input.")
    except Exception:
        st.info("This model does not support feature-level interpretability.")

  @st.cache_data
def generate_wordcloud(text):
    wc = WordCloud(width=400, height=200, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# Usage in main app flow
st.divider()
st.subheader("🎨 Key Word Map")
fig = generate_wordcloud(user_input)
st.pyplot(fig)


if st.session_state.analysis_done:
    st.subheader("User Feedback✍️")
    user_feedback = st.text_area("Please share any comments about usefulness or suggestions:", height=100)
    submit_feedback = st.button("📩 Submit Feedback")
    if submit_feedback and user_feedback.strip():
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

st.markdown("---")
st.markdown("""
   <div style="text-align: center; font-size: 0.85em; color: gray; line-height: 1.6em;">
    <strong>CaptAI™</strong>: Designed and Developed at <strong>AWC Labs</strong><br>
    📂 GitHub: <a href="https://github.com/Abdul-WriteCodes" target="_blank">AWC Labs</a><br>
    We appreciate voluntary support via 
    <a href="https://www.buymeacoffee.com/abdul_writecodes" target="_blank" style="color: #ff5f1f; font-weight: bold;">☕BuyMeACoffee</a> or 
    <a href="https://www.selar.com/showlove/awc-labs" target="_blank" style="color: #ff5f1f; font-weight: bold;">💖Selar</a><br>
    <strong>Disclaimer:</strong> We do not store personal data. Feedback is used only to improve the system.<br>
    © 2025 AWC Labs. All rights reserved.
</div>
""", unsafe_allow_html=True)