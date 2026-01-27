"""
Inference Module for XGBoost Pipeline.
"""
import joblib
import xgboost as xgb
import json
from pathlib import Path
from src.data.preprocess import clean_text

# Pointing to the new artifacts folder
ARTIFACTS_DIR = Path("artifacts_xgb") / "final_tfidf_xgboost_aug"

def load_artifacts():
    vectorizer = joblib.load(ARTIFACTS_DIR / "vectorizer.pkl")
    encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    model = xgb.XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "model.json")
    with open(ARTIFACTS_DIR / "label_map.json", "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return model, vectorizer, encoder, label_map

def predict_xgboost(texts, confidence_threshold=0.50):
    model, vectorizer, encoder, label_map = load_artifacts()
    cleaned_texts = [clean_text(t) for t in texts]
    X_input = vectorizer.transform(cleaned_texts)
    
    probs_matrix = model.predict_proba(X_input)
    predictions = probs_matrix.argmax(axis=1)
    
    results = []
    for text, pred_idx, probs in zip(texts, predictions, probs_matrix):
        confidence = float(probs.max())
        status = "accepted" if confidence >= confidence_threshold else "uncertain"
        class_probs = {label_map[i]: float(p) for i, p in enumerate(probs)}
        
        results.append({
            "text": text,
            "prediction": label_map[pred_idx],
            "confidence": confidence,
            "status": status,
            "probabilities": class_probs
        })
    return results