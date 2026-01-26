"""
Inference script: Load trained pipeline and make predictions on new texts.

This script loads the trained Logistic Regression + TF-IDF model and applies
the same preprocessing used during training before making predictions.

Usage:
    python scripts/infer.py
"""

import json
import joblib
from pathlib import Path
from src.data.preprocess import clean_text


# Path to trained artifacts
ARTIFACTS_DIR = Path("artifacts") / "final_tfidf_lr_aug"


def assert_artifacts_exist():
    """
    Check if all required artifact files exist.
    Raises FileNotFoundError with helpful message if any are missing.
    """
    required = ["model.pkl", "vectorizer.pkl", "config.json", "label_map.json"]
    missing = [p for p in required if not (ARTIFACTS_DIR / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {ARTIFACTS_DIR}: {missing}\n"
            f"Train first by running: python scripts/train_save_final.py"
        )


def load_artifacts():
    """
    Load trained model and vectorizer from disk.
    Checks that artifacts exist before attempting to load.
    
    Returns:
        Tuple of (model, vectorizer, config)
    """
    assert_artifacts_exist()
    
    model_path = ARTIFACTS_DIR / "model.pkl"
    vectorizer_path = ARTIFACTS_DIR / "vectorizer.pkl"
    config_path = ARTIFACTS_DIR / "config.json"
    label_map_path = ARTIFACTS_DIR / "label_map.json"
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    with open(label_map_path, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    
    return model, vectorizer, config, label_map 


def predict(texts, model=None, vectorizer=None, confidence_threshold=0.50):
    """
    Load artifacts (if not provided) and make predictions on new texts.
    Applies the same preprocessing (clean_text) used during training.
    
    Args:
        texts: List of text strings to classify
        model: Optional pre-loaded model. If None, will load from disk.
        vectorizer: Optional pre-loaded vectorizer. If None, will load from disk.
        confidence_threshold: Float between 0 and 1. Predictions with confidence
                            >= threshold are marked "accepted", < threshold are "uncertain".
                            Default 0.50 (50%).
        
    Returns:
        Tuple of (results, model, vectorizer) where results is a list of dicts with:
            - 'text': original text (truncated to 100 chars if necessary)
            - 'truncated': boolean indicating if text was truncated
            - 'prediction': predicted label (class index)
            - 'confidence': max probability
            - 'status': "accepted" (confidence >= threshold) or "uncertain" (< threshold)
            - 'probabilities': dict of {class_label: probability}
    """
    if model is None or vectorizer is None:
        model, vectorizer, _, label_map = load_artifacts()
    else:
        _, _, _, label_map = load_artifacts()
    
    # Apply same preprocessing as training data
    cleaned_texts = [clean_text(text) for text in texts]
    X = vectorizer.transform(cleaned_texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Format output with confidence and probabilities
    results = []
    for orig_text, pred, probs in zip(texts, predictions, probabilities):
        text_display = orig_text[:100]
        truncated = len(orig_text) > 100
        confidence = float(probs.max())
        
        # Determine status based on confidence threshold
        if confidence >= confidence_threshold:
            status = "accepted"
        else:
            status = "uncertain"
        
        # Use model.classes_ for robust class label mapping
        prob_dict = {int(cls): float(p) for cls, p in zip(model.classes_, probs)}
        
        results.append({
            "text": text_display,
            "truncated": truncated,
            "prediction_id": int(pred),
            "prediction_label": label_map[int(pred)],
            "confidence": confidence,
            "status": status,
            "probabilities": {
                label_map[int(cls)]: float(p)
                for cls, p in prob_dict.items()
        }
    })

    
    return results, model, vectorizer


def print_results(results, top_k=3):
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        text_display = r["text"] + ("..." if r["truncated"] else "")
        print(
        f"\n[{i}] ({r['status']})  "
        f"Pred: {r['prediction_label']} "
        f"(id={r['prediction_id']})  "
        f"Conf: {r['confidence']:.4f}"
        )

        print(f"     Text: {text_display}")

        # sort probs desc and show top-k
        probs_sorted = sorted(r["probabilities"].items(), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"     Top-{top_k} classes:")
        for label, p in probs_sorted:
            print(f"       {label}: {p:.4f}")




if __name__ == "__main__":
    # Example inference
    test_texts = [
        "This movie was absolutely amazing! I loved every second of it.",
        "I hated this film. It was boring and predictable.",
        "The acting was okay but the plot was confusing.",
        "I'm not sure how I feel about this movie."
    ]
    
    print("Loading trained pipeline...")
    model, vectorizer, config, label_map = load_artifacts()
    print(f" Model loaded: {config['model_type']} with {config['feature_extractor']}")
    
    print(f"\nMaking predictions on {len(test_texts)} texts...")
    results, _, _ = predict(test_texts, model=model, vectorizer=vectorizer)
    
    print_results(results)
    
    # Optionally save results to JSON
    output_file = Path("artifacts") / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to {output_file}")
