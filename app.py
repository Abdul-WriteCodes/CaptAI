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




# Generate a unique session ID for the user
session_id = str(uuid.uuid4())

# Google Form POST URL
form_url = "https://docs.google.com/forms/u/0/d/1LbiPqyGcf5dSjF7CntQQNN9HloumgcncQ-Rj2xCNFrk/formResponse"

# Integration of Google Form: Replace with  entry.xxxxx IDs from Google Form
ENTRY_SESSION = "entry.153544605"
ENTRY_LOG = "entry.588008680"
ENTRY_FEEDBACK = "entry.2051031519"

# Page Setup
st.set_page_config(page_title="🎭CaptAI", layout="centered")
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



@st.cache_resource
def load_models():
    almirax_model = joblib.load('almirax_pipeline.pkl')
    alekxia_model = joblib.load('alekxia_pipeline.pkl')
    return {
        'Almirax': almirax_model,
        'Alekxia': alekxia_model
    }

models = load_models()

with st.sidebar:
    dark_mode = st.checkbox("🌙 Dark Mode", value=False)

def set_theme(dark_mode: bool):
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        textarea, input, select {
            background-color: #22252a !important;
            color: white !important;
        }
        button {
            background-color: #444 !important;
            color: white !important;
        }
        [data-testid="stSidebar"] {
            background-color: #111518;
            color: white;
        }
        .stDivider {
            border-color: #444 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

set_theme(dark_mode)

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

# Initialize session state
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# Input Section
st.subheader("📝 Analyze Sentiment from Text Review")
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
        if prediction == 1:
            st.success("This review is Positive! 🎉", icon="👍")
            review_type = "**Positive**"
        else:
            st.error("This review is Negative. 😔", icon="👎")
            review_type = "**Negative**"

        st.metric(f"Hi, I am {model_choice}, my confidence about this {review_type} Review is:", f"{confidence*100:.1f}%")

        st.session_state.analysis_done = True
        st.session_state.user_log = f"Used model {model_choice}, prediction: {review_type}, confidence: {confidence:.2f}, input: {user_input[:100]}"

        # Feature Impact (only works if model has coef_ and TfidfVectorizer)
        try:
            vectorizer = model.named_steps['vectorizer']
            feature_names = vectorizer.get_feature_names_out()
            input_vector = model.named_steps['vectorizer'].transform([user_input])
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
        except Exception as e:
            st.info("This model does not support feature-level interpretability.")

        st.divider()

        # Word Cloud
        @st.cache_data
        def generate_wordcloud(text):
            wc = WordCloud(width=400, height=200, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            return fig

        st.subheader("🎨 Key Word Map")
        st.markdown(f"Here is the visual representation of the most prominent words in the text I analyzed to be **{review_type}**:")
        fig = generate_wordcloud(user_input)
        st.pyplot(fig)
        st.markdown("👩‍🔬I hope you find all of these information to be helpful ")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Feedback Section
if st.session_state.analysis_done:
    with st.expander("✍️ Submit Feedback (optional)"):
        user_feedback = st.text_area("We'd love your thoughts! How useful was this? Any suggestions?", height=100)
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
