"""
Final pipeline: SVM + TF-IDF on non augmented training data.
Trains the model and saves all artifacts (model, vectorizer, metrics, config).
"""

import os
import json
import joblib
from pathlib import Path

from sklearn.svm import  SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.data.train_dataset_EDA import load_train_with_features, load_test_with_features
from src.features.tfidf import fit_tfidf_on_any_dataframe
from src.data.augmentation import AugmentConfig, class_balanced_augment


# Artifact directory, unique for the inference of each model
ARTIFACTS_DIR = Path("artifacts_svm") / "final_tfidf_svm_aug"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


def save_artifacts(model, vectorizer, metrics, config, label_map):
    """
    copied the function that saves trained model, vectorizer, metrics, and config to disk
    but to a different path using joblib/JSON.

    Args:
        model: Trained SVM model
        vectorizer: Fitted TfidfVectorizer
        metrics: Dictionary with evaluation metrics
        config: Dictionary with training configuration
    """
    # Save model
    model_path = ARTIFACTS_DIR / "model.pkl"
    joblib.dump(model, model_path)
    print(f" Model saved to {model_path}")

    # Save vectorizer
    vectorizer_path = ARTIFACTS_DIR / "vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f" Vectorizer saved to {vectorizer_path}")

    # Save metrics
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f" Metrics saved to {metrics_path}")

    # Save config
    config_path = ARTIFACTS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Save label map
    label_map_path = ARTIFACTS_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {label_map_path}")


def load_artifacts():
    """
    Load trained model and vectorizer from disk using joblib.

    Returns:
        Tuple of (model, vectorizer)
    """
    model_path = ARTIFACTS_DIR / "model.pkl"
    vectorizer_path = ARTIFACTS_DIR / "vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def train_and_save():
    """
    Train SVC (linear, C=10) + TF-IDF on data and save artifacts.
    """
    print("=" * 70)
    print("Training: SVC + TF-IDF")
    print("=" * 70)

    print("\n[1/5] Loading training and test data...")
    train_df = load_train_with_features()
    test_df = load_test_with_features()
    print(f"  Original train size: {len(train_df)}")
    print(f"  Test size: {len(test_df)}")

    print("\n[2/5] Augmenting training data (class-balanced EDA)...")
    aug_cfg = AugmentConfig(
        mode="eda",
        n_aug_per_sample=1,
        keep_original=True,
        seed=42
    )

    train_df_aug = class_balanced_augment(
        df=train_df,
        text_col="clean_text",
        label_col="label",
        target_per_class=None,
        config=aug_cfg
    )

    print(f"  Augmented train size: {len(train_df_aug)}")
    print(f"  Original label counts:\n{train_df['label'].value_counts().sort_index().to_string()}")
    print(f"  Augmented label counts:\n{train_df_aug['label'].value_counts().sort_index().to_string()}")

    # Fit TF-IDF vectorizer on AUGMENTED training data
    print("\n[2/5] Fitting TF-IDF vectorizer on original training data...")
    vectorizer, X_train, y_train = fit_tfidf_on_any_dataframe(
        df=train_df_aug,
        text_col="clean_text",
        label_col="label",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000
    )
    print(f"  TF-IDF feature shape: {X_train.shape}")
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Transform test data using fitted vectorizer
    print("\n[3/5] Transforming test data...")
    X_test = vectorizer.transform(test_df["clean_text"])
    y_test = test_df["label"].values
    print(f"  Test feature shape: {X_test.shape}")

    print("\n[4/5] Training SVC (linear kernel, C=10.0, probability=True)...")
    model = SVC(
        kernel="linear",
        C=10.0,
        probability=True,
        class_weight=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("  Model training complete!")

    print("\n[5/5] Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, digits=4))

    print("=" * 70)
    print("CONFUSION MATRIX (TEST SET)")
    print("=" * 70)
    print(confusion_matrix(y_test, y_pred))

    # Prepare metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
        "augmented_train_size": int(len(train_df_aug)),
        "original_train_size": int(len(train_df)),
        "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
        "feature_shape": list(X_test.shape),
        "classification_report": classification_report(
            y_test, y_pred, digits=4, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Prepare config dictionary
    config = {
        "model_type": "SVC",
        "feature_extractor": "TfidfVectorizer",
        "augmentation": {
            "mode": aug_cfg.mode,
            "n_aug_per_sample": aug_cfg.n_aug_per_sample,
            "keep_original": aug_cfg.keep_original,
            "seed": aug_cfg.seed
        },
        "tfidf_params": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 20000
        },
        "svc_params": {
            "kernel": "linear",
            "C": 10.0,
            "probability": True,
            "class_weight": None,
            "random_state": 42
        }
    }

    # Save all artifacts
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)
    save_artifacts(model, vectorizer, metrics, config, LABEL_MAP)

    print("\n✓ Training pipeline complete!")
    print(f"✓ All artifacts saved to: {ARTIFACTS_DIR.resolve()}")

    return model, vectorizer, metrics


def predict(texts, confidence_threshold=0.50):
    """
    Load artifacts and make predictions on new texts.
    Applies the same preprocessing (clean_text) used during training.
    """
    from src.data.preprocess import clean_text

    model, vectorizer = load_artifacts()

    cleaned_texts = [clean_text(text) for text in texts]
    X = vectorizer.transform(cleaned_texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = []
    for orig_text, pred, probs in zip(texts, predictions, probabilities):
        confidence = float(probs.max())
        status = "accepted" if confidence >= confidence_threshold else "uncertain"

        prob_dict = {int(cls): float(p) for cls, p in zip(model.classes_, probs)}

        results.append({
            "text": orig_text,
            "prediction": int(pred),
            "confidence": confidence,
            "status": status,
            "probabilities": {
                LABEL_MAP[int(cls)]: float(p)
                for cls, p in prob_dict.items()
            }
        })

    return results


if __name__ == "__main__":
    model, vectorizer, metrics = train_and_save()
