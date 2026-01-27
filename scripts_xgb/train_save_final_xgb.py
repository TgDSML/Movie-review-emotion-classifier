"""
Final pipeline: XGBoost + TF-IDF on augmented training data.
Run from root: python -m scripts_xgb.train_save_final_xgboost
"""

import json
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.data.train_dataset_EDA import load_train_with_features, load_test_with_features
from src.features.tfidf import fit_tfidf_on_any_dataframe
from src.data.augmentation import AugmentConfig, class_balanced_augment

# Artifacts directory
ARTIFACTS_DIR = Path("artifacts_xgb") / "final_tfidf_xgboost_aug"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

def save_artifacts(model, vectorizer, encoder, metrics, config, label_map):
    # Save XGBoost Model
    model.save_model(ARTIFACTS_DIR / "model.json")
    print(f" Model saved to {ARTIFACTS_DIR / 'model.json'}")
    
    # Save Vectorizer & Encoder
    joblib.dump(vectorizer, ARTIFACTS_DIR / "vectorizer.pkl")
    joblib.dump(encoder, ARTIFACTS_DIR / "label_encoder.pkl")
    
    # Save Metrics, Config, Label Map
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(ARTIFACTS_DIR / "config.json", "w") as f: json.dump(config, f, indent=2)
    with open(ARTIFACTS_DIR / "label_map.json", "w") as f: json.dump(label_map, f, indent=2)

def train_and_save():
    print("=" * 70)
    print(" Training: XGBoost + TF-IDF (Augmented Pipeline)")
    print("=" * 70)
    
    # 1. Load & Augment
    train_df = load_train_with_features()
    test_df = load_test_with_features()
    aug_cfg = AugmentConfig(mode="eda", n_aug_per_sample=1, keep_original=True, seed=42)
    print("[1/4] Augmenting Data...")
    train_df_aug = class_balanced_augment(train_df, 'clean_text', 'label', config=aug_cfg)
    
    # 2. Fit TF-IDF
    print("[2/4] Fitting TF-IDF...")
    vectorizer, X_train, y_train = fit_tfidf_on_any_dataframe(
        train_df_aug, 'clean_text', 'label', ngram_range=(1, 2), max_features=20000
    )
    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df['label'].values

    # 3. Encode Labels & Train
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("[3/4] Training XGBoost...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=len(le.classes_),
        n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train_enc)
    
    # 4. Evaluate
    print("[4/4] Evaluating...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"🏆 Test Accuracy: {acc:.2%}")
    print(classification_report(y_test_enc, preds, digits=4))
    
    metrics = {
        "accuracy": acc,
        "classification_report": classification_report(y_test_enc, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test_enc, preds).tolist()
    }
    config = {"model": "XGBoost", "n_estimators": 100, "max_depth": 6, "augmentation": "EDA"}

    save_artifacts(model, vectorizer, le, metrics, config, LABEL_MAP)
    print("\n Pipeline Complete!")

if __name__ == "__main__":
    train_and_save()