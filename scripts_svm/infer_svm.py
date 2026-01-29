import json
import joblib
from pathlib import Path
from src.data.preprocess import clean_text

ARTIFACTS_DIR = Path("artifacts_svm") / "final_tfidf_svm_aug"
def assert_artifacts_exist():
    required = ["model.pkl", "vectorizer.pkl", "config.json", "label_map.json"]
    missing = [p for p in required if not (ARTIFACTS_DIR / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {ARTIFACTS_DIR}: {missing}\n"
            f"Train first by running: python scripts/train_save_final_svm.py"
        )


def load_artifacts():
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
def predict_svm(texts, confidence_threshold=0.80):
    model, vectorizer, config, label_map = load_artifacts()

    cleaned_texts = [clean_text(t) for t in texts]
    X_input = vectorizer.transform(cleaned_texts)

    probs_matrix = model.predict_proba(X_input)
    class_ids = model.classes_                      
    pred_class_ids = class_ids[probs_matrix.argmax(axis=1)]

    results = []
    for text, pred_cls, probs in zip(texts, pred_class_ids, probs_matrix):
        confidence = float(probs.max())
        status = "accepted" if confidence >= confidence_threshold else "uncertain"

        class_probs = {
            label_map[int(cls)]: float(p)
            for cls, p in zip(class_ids, probs)
        }

        results.append({
            "text": text,
            "prediction_id": int(pred_cls),
            "prediction_label": label_map[int(pred_cls)],
            "confidence": confidence,
            "status": status,
            "probabilities": class_probs
        })

    return results
if __name__ == "__main__":
    test_texts = [
        "This movie was absolutely amazing! I loved every second of it.",
        "I hated this film. It was boring and predictable.",
        "The acting was okay but the plot was confusing.",
        "I'm not sure how I feel about this movie."
    ]

    print("Loading trained pipeline...")
    model, vectorizer, config, label_map = load_artifacts()
    print(f"Model loaded: {config['model_type']} with {config['feature_extractor']}")

    print(f"\nMaking predictions on {len(test_texts)} texts...")
    results = predict_svm(test_texts, confidence_threshold=0.80)

    for r in results:
        print(f"\n({r['status']}) Pred: {r['prediction_label']}  Conf: {r['confidence']:.4f}")
        print(f"Text: {r['text']}")

    output_file = ARTIFACTS_DIR / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

