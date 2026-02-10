"""
Inference Module for XGBoost Pipeline.
FIXED VERSION: Correct paths + Hardcoded Label Map
"""
import joblib
import xgboost as xgb
import json
from pathlib import Path
from src.data.preprocess import clean_text  # Χρησιμοποιούμε τη δική σου συνάρτηση καθαρισμού

# --- ΣΩΣΤΟ ΜΟΝΟΠΑΤΙ (Βάσει του κώδικα που μου έστειλες) ---
ARTIFACTS_DIR = Path("artifacts_xgb") / "final_tfidf_xgboost_aug"

# --- HARDCODED LABEL MAP ---
# Το γράφουμε εδώ ρητά για να είμαστε σίγουροι ότι το 1 είναι JOY και όχι Surprise
FIXED_LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def load_artifacts():
    # Φόρτωση του vectorizer (με το σωστό όνομα 'vectorizer.pkl')
    vectorizer = joblib.load(ARTIFACTS_DIR / "vectorizer.pkl")
    
    # Φόρτωση του Label Encoder (αν και δεν το χρησιμοποιούμε στο hardcoded map, το φορτώνουμε για τυπικούς λόγους)
    # Αν βγάλει error εδώ, διέγραψε αυτή τη γραμμή
    encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    
    # Φόρτωση του XGBoost μοντέλου
    model = xgb.XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "model.json")
    
    return model, vectorizer, encoder

def predict_xgboost(texts, confidence_threshold=0.50):
    # 1. Φόρτωση
    model, vectorizer, encoder = load_artifacts()
    
    # 2. Καθαρισμός (Πολύ σημαντικό: χρησιμοποιούμε το δικό σου clean_text)
    cleaned_texts = [clean_text(t) for t in texts]
    
    # 3. Μετατροπή σε νούμερα (Vectorization)
    X_input = vectorizer.transform(cleaned_texts)
    
    # Debug Print: Πόσες λέξεις αναγνώρισε;
    # Αν δεις 0 εδώ, σημαίνει ότι ο clean_text σβήνει τα πάντα ή οι λέξεις δεν υπάρχουν
    print(f"DEBUG Check: Input features (non-zero): {X_input.nnz}")

    # 4. Πρόβλεψη
    probs_matrix = model.predict_proba(X_input)
    predictions = probs_matrix.argmax(axis=1)
    
    results = []
    for text, pred_idx, probs in zip(texts, predictions, probs_matrix):
        confidence = float(probs.max())
        status = "accepted" if confidence >= confidence_threshold else "uncertain"
        
        # ΧΡΗΣΗ ΤΟΥ HARDCODED MAP
        prediction_label = FIXED_LABEL_MAP.get(int(pred_idx), "Unknown")
        
        # Φτιάχνουμε και τα probabilities με το σωστό map
        class_probs = {FIXED_LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
        
        results.append({
            "text": text,
            "prediction": prediction_label, # Εδώ μπαίνει το σωστό label
            "confidence": confidence,
            "status": status,
            "probabilities": class_probs
        })
        
    return results

if __name__ == "__main__":
    # Γρήγορο τεστ αν τρέξεις το αρχείο μόνο του
    test = ["I am so happy and full of joy"]
    print(predict_xgboost(test))