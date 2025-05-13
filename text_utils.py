# text_utils.py

import re

def text_cleaning(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def batch_text_cleaning(text_list):
    return [text_cleaning(t) for t in text_list]
