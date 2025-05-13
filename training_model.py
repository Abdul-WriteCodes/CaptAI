# train_pipelines.py
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:47:39 2025
@author: HP
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Import top-level text cleaning functions
from text_utils import batch_text_cleaning, text_cleaning

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\IMDB%20Dataset.csv")

# Clean the text once for dataset quality
df["review"] = df["review"].apply(text_cleaning)

# Encode sentiment labels
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split features and labels
X = df["review"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use a named, importable function for FunctionTransformer
text_cleaner = FunctionTransformer(batch_text_cleaning, validate=False)

# ---- SGDClassifier pipeline ----
almirax_pipeline = Pipeline([
    ("cleaner", text_cleaner),
    ("vectorizer", TfidfVectorizer(max_features=5000)),
    ("classifier", SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, random_state=42))
])
almirax_pipeline.fit(X_train, y_train)
joblib.dump(almirax_pipeline, "almirax_pipeline.pkl")

# ---- LogisticRegression pipeline ----
alekxia_pipeline = Pipeline([
    ("cleaner", text_cleaner),
    ("vectorizer", TfidfVectorizer(max_features=5000)),
    ("classifier", LogisticRegression())
])
alekxia_pipeline.fit(X_train, y_train)
joblib.dump(alekxia_pipeline, "alekxia_pipeline.pkl")

print("✅ Pipelines trained and saved: almirax_pipeline.pkl and alekxia_pipeline.pkl")
