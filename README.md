# 🎬 Movie Review Emotion Classifier

This repository contains a **machine learning project for emotion classification in text**, with a focus on **movie reviews and social media comments**. Unlike classic sentiment analysis (positive/negative), this project aims to predict **fine‑grained emotions** such as *joy, sadness, anger, fear,* etc.

The project is developed in a **research‑style, modular way**, combining exploratory notebooks with reusable Python modules.

---

## 📌 Project Objectives

* Perform **emotion classification** on text data
* Compare **different text representations**:

  * TF‑IDF
  * Word embeddings like Word2Vec and Glove
* Train and evaluate **classical machine learning models** (Logistic Regression)
* Build a **clean, reproducible ML pipeline** suitable for academic work

---

## 📂 Repository Structure

```
Movie-review-emotion-classifier/
│
├── data/
│   ├── *_train.csv
│   ├── *_test.csv
│   └── (processed datasets generated during experiments)
│
├── notebooks/
│   ├── analysis.ipynb
│   ├── logistic_regression.ipynb
│   ├── tfidf_intuition.ipynb
│   └── (exploratory & experimental notebooks)
│
├── src/
│   ├── data/
│   │   ├── EDA_visual.py
│   │   ├── feature_extraction.py
│   │   └── preprocessing utilities
│   │
│   ├── models/
│   │   ├── logistic_regression.py
│   │   └── model-related logic
│   │
│   ├── utils/
│   │   └── helper scripts (paths, loading, inspection)
│   │
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Notebooks Overview

### `analysis.ipynb`

* End‑to‑end experimental notebook
* Data loading and cleaning
* Feature engineering
* Model training & evaluation
* Confusion matrices and per‑class metrics

### `logistic_regression.ipynb`

* Logistic Regression experiments
* TF‑IDF vs GloVe comparison
* Detailed evaluation outputs

### `tfidf_intuition.ipynb`

* Educational notebook
* Explains how TF‑IDF works intuitively
* Used for understanding feature behavior

---

## 🧠 Models Implemented

### Logistic Regression

Used as a strong baseline for text classification:

* Works with **TF‑IDF vectors**
* Works with **sentence‑level GloVe embeddings** (mean pooling)

Evaluation includes:

* Accuracy
* Precision / Recall / F1 (per class)
* Confusion matrices

---

## 🔠 Text Representations

### TF‑IDF

* Sparse, high‑dimensional
* Strong baseline for classical NLP
* Typically outperforms simple averaged embeddings

### GloVe Embeddings

* Pretrained dense word vectors
* Sentence vectors obtained via **mean pooling**
* Lower dimensional but may lose contextual nuance

---

## 📊 Evaluation Strategy

* Train / test split performed **once** and reused across models
* Vectorizers and embeddings are **fit only on training data**
* Test data is strictly unseen
* Evaluation focuses on **per‑emotion performance**, not just accuracy

---

## ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/TgDSML/Movie-review-emotion-classifier.git
cd Movie-review-emotion-classifier

# Create virtual environment
python -m venv .venv

# Activate
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run notebooks

Open VS Code or Jupyter and run notebooks from the `notebooks/` directory.

### Run scripts

From project root:

```bash
python src/utils/inspect.py
```

---

## 📈 Project Status

✅ Data preprocessing complete
✅ TF‑IDF + Logistic Regression implemented
✅ GloVe + Logistic Regression implemented
✅ Detailed evaluation metrics added
🚧 Ongoing experimentation & improvements

---

## 📌 Future Improvements

* Advanced pooling strategies for embeddings
* Data augmentation for text
* Regularization & hyperparameter tuning

---



