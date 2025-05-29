import pytest
import joblib
import numpy as np
from wordcloud import WordCloud
import requests
from unittest.mock import patch

# Constants (Replace with actual values in app)
ENTRY_SESSION = "entry.153544605"
ENTRY_LOG = "entry.588008680"
ENTRY_FEEDBACK = "entry.2051031519"
form_url = "https://docs.google.com/forms/u/0/d/1LbiPqyGcf5dSjF7CntQQNN9HloumgcncQ-Rj2xCNFrk/formResponse"

# Load models
@pytest.fixture(scope="module")
def models():
    almirax = joblib.load("almirax_pipeline.pkl")
    alekxia = joblib.load("alekxia_pipeline.pkl")
    return {"Almirax": almirax, "Alekxia": alekxia}

# 1. Test model loading
def test_model_loading(models):
    assert "Almirax" in models
    assert "Alekxia" in models

# 2. Test prediction on simple text
def test_model_prediction(models):
    text = "This is the best product I have ever used!"
    for name, model in models.items():
        result = model.predict([text])[0]
        assert result in [0, 1]

# 3. Test predict_proba functionality
def test_model_predict_proba(models):
    text = "This app is terrible"
    for model in models.values():
        assert hasattr(model, "predict_proba")
        proba = model.predict_proba([text])[0]
        assert len(proba) == 2
        assert np.isclose(sum(proba), 1.0)

# 4. Test interpretability components
def test_model_interpretability(models):
    for model in models.values():
        if hasattr(model.named_steps["classifier"], "coef_"):
            coef = model.named_steps["classifier"].coef_
            assert coef.shape[0] == 1  # Binary classification

# 5. Test word cloud generation
def test_wordcloud_generation():
    text = "good bad average fantastic horrible excellent poor great nice awful"
    wc = WordCloud(width=400, height=200, background_color='white').generate(text)
    assert wc is not None
    assert hasattr(wc, "words_")

# 6. Test feedback submission with mock
@patch('requests.post')
def test_feedback_submission(mock_post):
    mock_post.return_value.status_code = 200
    form_data = {
        ENTRY_SESSION: 'test-session',
        ENTRY_LOG: 'Test log',
        ENTRY_FEEDBACK: 'This app is super helpful!'
    }
    response = requests.post(form_url, data=form_data)
    assert response.status_code == 200
