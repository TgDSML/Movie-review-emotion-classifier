#  Movie Review Emotion Classifier

Machine learning project for **emotion classification from text**, with emphasis on **movie reviews** and short social-media style texts.  
Unlike binary sentiment analysis (positive/negative), this project focuses on **fine-grained emotion recognition** across six emotional categories.

The repository is structured in a **research-oriented and reproducible way**, combining exploratory notebooks with reusable training and inference scripts.

---

##  Problem Description

Given a piece of text (e.g. a movie review), the goal is to classify it into one of the following **six emotion classes**:

- `sadness`
- `joy`
- `love`
- `anger`
- `fear`
- `surprise`

The task is formulated as a **multiclass text classification problem** under class imbalance.

---

##  Project Objectives

- Perform **emotion classification** on textual data
- Compare different **text representations**:
  - TF-IDF (unigrams & bigrams)
  - Word embeddings (Word2Vec, GloVe)
- Train and evaluate **classical machine learning models**
- Apply **data augmentation** to mitigate class imbalance
- Build a **clean, end-to-end ML pipeline** suitable for MSc-level coursework

---

##  Repository Structure

```
Movie-review-emotion-classifier/
│
├── data/ # raw & processed datasets
│ ├── emotion_train.csv
│ ├── emotion_test.csv
│ ├── emotion_processed_train.csv
│ └── emotion_processed_test.csv
│
├── notebooks/ # exploratory & demo notebooks
│ ├── analysis.ipynb
│ ├── logistic_regression.ipynb
│ ├── tfidf_intuition.ipynb
│ └── inference_demo.ipynb
│
├── scripts_lr/ # Logistic Regression training & inference
│ ├── train_save_final.py
│ ├── infer.py
│ └── infer_imdb.py
│
├── scripts_svm/ # SVM experiments
│ ├── train_save_final_svm.py
│ └── infer_svm.py
│
├── scripts_xgb/ # XGBoost experiments
│ ├── train_save_final_xgb.py
│ └── infer_xgb.py
│
├── src/ # reusable pipeline components
│ ├── data/
│ ├── features/
│ ├── models/
│ └── utils/
│
├── artifacts/ # generated after training
│ ├── model.pkl
│ ├── vectorizer.pkl
│ ├── label_map.json
│ ├── metrics.json
│ ├── config.json
│ └── predictions.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Notebooks Overview

The `notebooks/` directory contains exploratory, experimental and educational notebooks used during the development of the project.  
These notebooks are **not required** to run the training or inference pipelines, but they document the full research process.

### `analysis.ipynb`
- Main exploratory and experimental notebook
- Dataset inspection and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Confusion matrices and per-class metrics

### `augmentation.ipynb`
- Exploration of data augmentation techniques for text
- Synonym-based augmentation experiments
- Analysis of class balancing effects

### `logistic_regression.ipynb`
- Logistic Regression experiments
- TF-IDF with and without data augmentation
- Comparison with GloVe and Word2Vec embeddings
- Detailed classification reports and metrics

### `svm_with_tfidf.ipynb`
- Linear SVM experiments using TF-IDF features
- Evaluation of SVM performance on the emotion classification task

### `xgboost.ipynb`
- XGBoost experiments with TF-IDF features
- Performance comparison with linear models

### `tfidf_intuition.ipynb`
- Educational notebook explaining TF-IDF
- Step-by-step intuition and toy examples
- Used to understand feature weighting behavior

### `word2vec_intuition.ipynb`
- Educational notebook for Word2Vec embeddings
- Demonstrates how word vectors are formed and combined
- Analysis of sentence-level representations via pooling

### `glove_intuition_download_of_glove.6B.zip.ipynb`
- Notebook documenting the download and loading of pretrained GloVe embeddings
- Practical steps for integrating external embeddings
- MUST RUN in order to download from Stanford's Library the glove.6B.zip file which contains the word embeddings

### `glove_with_diff_class.ipynb`
- Experiments with GloVe embeddings across different emotion classes
- Analysis of embedding behavior per class

### `inference_demo.ipynb`
- Interactive inference demonstration
- Applies the trained model to custom movie reviews
- Displays predicted emotion labels and confidence scores


---

##  Models Implemented

- **Logistic Regression** (baseline & best performing)
- **Linear SVM**
- **XGBoost**

##  Feature Extraction

### TF-IDF
- Unigrams + bigrams
- Sparse, high-dimensional representation
- Strong baseline for classical NLP tasks

### Word Embeddings
- Pretrained **Word2Vec** and **GloVe**
- Sentence representations via pooling
- Lower dimensional but less expressive for this task

---
##  Evaluation Strategy

- Fixed **train/test split**
- All preprocessing, vectorization and augmentation applied **only on training data**
- Test set remains strictly unseen
- Main metric: **Macro-averaged F1-score**
  - Suitable for imbalanced multiclass classification

---

##  Best Performing Configuration for each model

**TF-IDF (unigrams + bigrams) + Logistic Regression + Data Augmentation**

- Accuracy: **0.8725**
- Macro-averaged F1-score: **0.8252**

** TF-IDF (unigrams + bigrams) + Linear SVM **

- Accuracy: **0.8590**
- Macro-averaged F1-score: **0.8009**

**TF-IDF (unigrams + bigrams) + XGBOOST + Data Augmentation**

- AccuracyQ **0.88**
- Macro-averaged F1-score: **0.850141**


##  Installation & Setup

# Clone repository
git clone https://github.com/TgDSML/Movie-review-emotion-classifier.git
cd Movie-review-emotion-classifier

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd)
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

---

###  How to Run

## Train Models

# Logistic Regression + TF-IDF + Augmentation
python scripts_lr/train_save_final.py

# Linear SVM + TF-IDF
python scripts_svm/train_save_final_svm.py

# XGBoost + TF-IDF + Augmentation
python scripts_xgb/train_save_final_xgb.py

---

##  Inference Demo on IMDb Reviews

# IMDb movie reviews (Logistic Regression)
python scripts_lr/infer_imdb.py

# SVM inference
python scripts_svm/infer_svm.py

# XGBoost inference
python scripts_xgb/infer_xgb.py

---

## Artifacts & Outputs

After training, the pipeline produces the following files:

- model.pkl – trained classifier  
- vectorizer.pkl – TF-IDF vectorizer  
- label_map.json – mapping between label ids and emotion names  
- metrics.json – evaluation metrics summary  
- config.json – training configuration  
- predictions.json – inference results produced during demo runs  

---

##  Reproducibility Notes

- Random seeds are fixed where applicable to ensure reproducibility  
- Data augmentation is applied only on the training set  
- Test data is never used during feature fitting or model selection  






