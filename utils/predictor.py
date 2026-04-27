"""
predictor.py
------------
Handles:
  1. Sentiment prediction using local RoBERTa model (sentiment_roberta/)
  2. Top-3 side effect extraction from drug reviews in the CSV dataset
"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "sentiment_roberta")
TRAIN_CSV  = os.path.join(BASE_DIR, "drugsComTrain_raw.csv")
TEST_CSV   = os.path.join(BASE_DIR, "drugsComTest_raw.csv")

# ── Common side-effect keywords to mine from reviews ─────────────────────────
SIDE_EFFECT_PATTERNS = re.compile(
    r'\b('
    r'nausea|vomiting|dizziness|headache|fatigue|drowsiness|insomnia|'
    r'diarrhea|constipation|rash|itching|swelling|weight gain|weight loss|'
    r'anxiety|depression|mood swings|dry mouth|blurred vision|hair loss|'
    r'stomach pain|abdominal pain|cramps|bloating|gas|heartburn|'
    r'chest pain|palpitations|shortness of breath|increased heart rate|'
    r'muscle pain|joint pain|back pain|weakness|tremors|'
    r'loss of appetite|increased appetite|sexual dysfunction|'
    r'hot flashes|sweating|fever|chills|infection|bleeding|bruising|'
    r'memory loss|confusion|hallucinations|suicidal thoughts|'
    r'kidney problems|liver problems|high blood pressure|low blood pressure'
    r')\b',
    re.IGNORECASE
)

# import torch
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# MODEL_DIR = r"C:\Users\MARUTIRAJ\Desktop\drug-side-effect-app\sentiment_roberta"

@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR,use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    print(model.config.id2label)
    model.eval()
    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu")
    model.to(device)
    return tokenizer,model,device

@lru_cache(maxsize=1)
def _load_dataset():
    dfs = []
    for path in [TRAIN_CSV, TEST_CSV]:
        if os.path.exists(path):
            df = pd.read_csv(
                path, encoding="utf-8", on_bad_lines="skip",
                usecols=["drugName", "review", "rating"]
            )
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("Dataset CSVs not found.")
    full = pd.concat(dfs, ignore_index=True)
    full["drugName_lower"] = full["drugName"].str.strip().str.lower()
    full["rating"]         = pd.to_numeric(full["rating"], errors="coerce")
    return full

def predict_sentiment(drug_name: str) -> dict:
    drug_name = drug_name.strip()
    dataset   = _load_dataset()
    key = drug_name.lower()
    drug_reviews = dataset[dataset["drugName_lower"] == key]
    found = len(drug_reviews) > 0

    if found:
        neg_reviews  = drug_reviews[drug_reviews["rating"] <= 5].sort_values("rating")
        text_for_clf = " ".join(neg_reviews["review"].dropna().head(5).tolist())[:512]
        if not text_for_clf:
            text_for_clf = drug_name
    else:
        text_for_clf = f"Patient review about the drug {drug_name}."

    tokenizer, model, device = _load_model()
    inputs = tokenizer(
        text_for_clf, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs     = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    label_map = model.config.id2label
    pred_idx  = int(np.argmax(probs))
    sentiment = label_map[pred_idx]
    confidence = float(probs[pred_idx])

    side_effects = []
    if sentiment == "Negative" and found:
        side_effects = _extract_top_side_effects(drug_reviews)

    return {
        "drug":          drug_name,
        "sentiment":     sentiment,
        "confidence":    round(confidence * 100, 1),
        "side_effects":  side_effects,
        "found_in_data": found,
    }

def _extract_top_side_effects(drug_reviews: pd.DataFrame, top_n: int = 3) -> list:
    low_rating = drug_reviews[drug_reviews["rating"] <= 5]["review"].dropna()
    counter = Counter()
    for review in low_rating:
        matches = SIDE_EFFECT_PATTERNS.findall(str(review).lower())
        counter.update(matches)
    if not counter:
        for review in drug_reviews["review"].dropna():
            matches = SIDE_EFFECT_PATTERNS.findall(str(review).lower())
            counter.update(matches)
    return [effect.title() for effect, _ in counter.most_common(top_n)]
