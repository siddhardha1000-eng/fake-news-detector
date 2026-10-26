# 📰 TruthScan: Fake News Detection System using Machine Learning

A complete, production-ready Machine Learning system for automated news authenticity classification, text preprocessing, multi-model evaluation, web scraping, and interactive prediction.

---

## 🚀 Key Modules & Architecture

```
fake_news_detector/
├── data/
│   ├── dataset_collector.py   # Dataset generator, benchmark loader & balancer
│   ├── news_dataset.csv       # Curated balanced dataset (Real: 0, Fake: 1)
│   └── scraper.py             # Live web scraper for news articles & URLs
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Cleaning, Tokenization, Lemmatization, TF-IDF N-grams
│   ├── models.py              # Logistic Regression, SVM, Random Forest, Naive Bayes
│   ├── evaluate.py            # Precision, Recall, F1, Confusion Matrix, ROC-AUC
│   └── pipeline.py            # End-to-end training and inference engine
├── models_saved/              # Serialized models (.joblib) & metrics (.json)
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── best_model.joblib
│   └── model_metrics.json
├── train.py                   # Script to train, evaluate, and save all models
├── predict.py                 # Command-line prediction tool with explainability
├── app.py                     # Streamlit web user interface
└── requirements.txt           # Project dependencies
```

---

## 🛠️ Implementation Details

### 1. Dataset Collection (`data/`)
- **Balanced Benchmark**: 1,200 curated and verified articles evenly split into **600 Authentic (Real)** and **600 Fabricated (Fake)** news items.
- **Topical Coverage**: Politics, Healthcare & Medicine, Science & Space Exploration, Technology, and World Affairs.
- **Web Scraper (`scraper.py`)**: Real-time article scraper capable of fetching titles, clean body text, and metadata from arbitrary news URLs while stripping ads, navigational clutter, and HTML tags.

### 2. Text Preprocessing (`src/preprocessing.py`)
- **Text Normalization**: Lowercasing, contraction expansion (`can't` -> `cannot`), URL stripping, HTML tag removal, non-alphabetic filtering.
- **Tokenization & Lemmatization**: Word tokenization with NLTK `word_tokenize`, English stopword elimination, and vocabulary normalization using NLTK `WordNetLemmatizer`.
- **Feature Extraction**: `TfidfVectorizer` with:
  - **N-grams**: `ngram_range=(1, 2)` (unigrams and bigrams like *"breaking news"*, *"clinical trial"*, *"miracle cure"*).
  - **Sublinear TF Scaling**: `sublinear_tf=True` (`1 + log(tf)`).
  - **Vocabulary Size**: 1,632 active n-gram features.
- **Linguistic Explainability**: Extracts highest-scoring n-gram tokens per article to explain why a model flagged an article as real or fake.

### 3. Classification Models (`src/models.py`)
- **Logistic Regression**: Linear classifier with L2 regularization and balanced class weights.
- **Support Vector Machine (SVM)**: `LinearSVC` wrapped in `CalibratedClassifierCV` to provide calibrated prediction probabilities.
- **Random Forest Classifier**: Ensemble of 150 bagged decision trees with feature subsampling.
- **Multinomial Naive Bayes**: Probabilistic word frequency baseline.

### 4. Model Evaluation & Comparison (`src/evaluate.py`)
Evaluated on a held-out 20% stratified test set ($N=240$ articles):

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Support Vector Machine** | **0.9958** | **0.9917** | **1.0000** | **0.9959** | **0.9958** | **1.0000** |

- **Metrics Computed**: Accuracy, Precision, Recall, F1-Score (Binary, Macro, Weighted), Confusion Matrix ($TN, FP, FN, TP$), and ROC-AUC.

---

## 💻 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train & Evaluate Models
To recreate the dataset, preprocess text, fit all classifiers, and compute test metrics:
```bash
python train.py
```

### Step 3: Run Interactive Web Interface (Streamlit)
Launch the web UI:
```bash
streamlit run app.py
```
The interface opens in your browser at `http://localhost:8501`. Features include:
1. **News Verification**: Paste text or pick pre-loaded authentic/fake news stories to get live verdicts, probability meters, and top indicative n-grams.
2. **Multi-Model Comparison**: Side-by-side metric charts, interactive confusion matrices, and classification reports.
3. **Live Web Scraper**: Paste any news URL (e.g. Wikipedia, BBC, Reuters) to scrape, extract, and verify directly.
4. **NLP Preprocessing Lab**: Inspect raw text $\to$ cleaned text $\to$ tokens $\to$ lemmatized tokens $\to$ TF-IDF weights step-by-step.

### Step 4: Run CLI Prediction Tool
Quickly test any headline from your terminal:
```bash
# Test single headline with the best model
python predict.py "James Webb Space Telescope discovers carbon dioxide on exoplanet"

# Compare predictions across all trained models
python predict.py "Grandmother cures all cancer in 3 days using miracle dandelion tea" --compare

# Interactive prompt mode
python predict.py --interactive
```
