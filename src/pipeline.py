"""
End-to-End Training and Inference Pipeline
Manages:
- Dataset loading and preprocessing
- TF-IDF feature extraction (n-grams)
- Multi-model training and evaluation
- Model persistence (joblib)
- Inference engine for real-time predictions
"""

import os
import time
import json
from typing import Dict, Any, Tuple, Optional, List
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    preprocess_text,
    preprocess_dataframe,
    create_tfidf_vectorizer,
    get_top_features_for_text
)
from src.models import get_models
from src.evaluate import compare_models, print_comparison_table
from data.dataset_collector import load_dataset, create_comprehensive_dataset


class PipelineTrainer:
    """Manages training, evaluation, comparison, and export of all models."""

    def __init__(
        self,
        dataset_path: str = "data/news_dataset.csv",
        models_dir: str = "models_saved",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.test_size = test_size
        self.random_state = random_state

        os.makedirs(self.models_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Executes full training and evaluation workflow."""
        print("\n[Step 1/5] Loading News Dataset...")
        if not os.path.exists(self.dataset_path):
            create_comprehensive_dataset(self.dataset_path)
        
        df = pd.read_csv(self.dataset_path)
        print(f"Loaded {len(df)} records. Real: {(df['label'] == 0).sum()} | Fake: {(df['label'] == 1).sum()}")

        print("\n[Step 2/5] Preprocessing and Tokenizing Text (Stopwords, Lemmatization)...")
        start_prep = time.time()
        # Combine title and text for stronger contextual representation
        combined_texts = df["title"].fillna("") + " " + df["text"].fillna("")
        cleaned_corpus = combined_texts.apply(preprocess_text)
        labels = df["label"].values
        print(f"Preprocessing completed in {time.time() - start_prep:.2f}s")

        print("\n[Step 3/5] Splitting Train/Test and Computing TF-IDF (Unigrams + Bigrams)...")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            cleaned_corpus,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )

        vectorizer = create_tfidf_vectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        X_train_tfidf = vectorizer.fit_transform(X_train_raw)
        X_test_tfidf = vectorizer.transform(X_test_raw)
        print(f"TF-IDF Vocabulary size: {len(vectorizer.vocabulary_)} features")
        print(f"Train samples: {X_train_tfidf.shape[0]}, Test samples: {X_test_tfidf.shape[0]}")

        print("\n[Step 4/5] Training Classification Models (Logistic Regression, SVM, Random Forest)...")
        models = get_models(random_state=self.random_state)
        trained_models = {}
        training_times = {}

        for name, model in models.items():
            t0 = time.time()
            print(f"  -> Training {name}...")
            model.fit(X_train_tfidf, y_train)
            dur = time.time() - t0
            training_times[name] = round(dur, 3)
            trained_models[name] = model
            print(f"     Finished in {dur:.2f}s")

        print("\n[Step 5/5] Evaluating and Comparing Multiple Models...")
        comp_df, detailed_metrics, best_model_name = compare_models(
            trained_models,
            X_test_tfidf,
            y_test
        )
        print_comparison_table(comp_df, best_model_name)

        # Save artifacts
        print(f"\nSaving trained models and vectorizer to '{self.models_dir}'...")
        joblib.dump(vectorizer, os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))

        model_filenames = {
            "Logistic Regression": "logistic_regression.joblib",
            "Support Vector Machine": "svm.joblib",
            "Random Forest": "random_forest.joblib",
            "Multinomial Naive Bayes": "naive_bayes.joblib"
        }

        for name, model in trained_models.items():
            fname = model_filenames.get(name, f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, os.path.join(self.models_dir, fname))

        # Save best model copy
        joblib.dump(trained_models[best_model_name], os.path.join(self.models_dir, "best_model.joblib"))

        # Save metadata and metrics
        metadata = {
            "best_model_name": best_model_name,
            "training_times": training_times,
            "comparison_summary": comp_df.to_dict(orient="records"),
            "detailed_metrics": detailed_metrics,
            "vocab_size": len(vectorizer.vocabulary_),
            "total_samples": len(df),
            "test_samples": int(len(y_test))
        }

        with open(os.path.join(self.models_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("Pipeline training successfully completed!")
        return metadata


class FakeNewsPredictor:
    """
    Inference Engine for predicting Real vs Fake news on user inputs.
    Loads saved vectorizer and models.
    """

    def __init__(self, models_dir: str = "models_saved"):
        self.models_dir = models_dir
        self.vectorizer = None
        self.models = {}
        self.best_model_name = "Logistic Regression"
        self.metadata = {}
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Loads serialized models, vectorizer, and metrics."""
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.models_dir}'. Please train the pipeline first."
            )

        self.vectorizer = joblib.load(vec_path)

        # Load metrics metadata
        meta_path = os.path.join(self.models_dir, "model_metrics.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.best_model_name = self.metadata.get("best_model_name", "Logistic Regression")

        # Load each model
        model_files = {
            "Logistic Regression": "logistic_regression.joblib",
            "Support Vector Machine": "svm.joblib",
            "Random Forest": "random_forest.joblib",
            "Multinomial Naive Bayes": "naive_bayes.joblib",
            "Best Model": "best_model.joblib"
        }

        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

    def predict(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predicts authenticity of a news article or headline.

        Args:
            text: Raw article body or headline
            model_name: Model to use. Defaults to the best model.

        Returns:
            Dict containing:
                - prediction: 'REAL' or 'FAKE'
                - label: 0 or 1
                - confidence: Probability score (0.0 - 1.0)
                - probabilities: {'REAL': float, 'FAKE': float}
                - top_keywords: Top indicative TF-IDF n-grams
                - model_used: Name of the model applied
        """
        if not text or not text.strip():
            return {
                "prediction": "UNKNOWN",
                "label": -1,
                "confidence": 0.0,
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "top_keywords": [],
                "model_used": model_name or self.best_model_name,
                "error": "Empty text provided"
            }

        target_model_name = model_name if (model_name and model_name in self.models) else "Best Model"
        model = self.models.get(target_model_name)
        if model is None:
            # Fallback to any loaded model
            model = next(iter(self.models.values()))

        # Preprocess input text
        processed = preprocess_text(text)
        tfidf_vec = self.vectorizer.transform([processed])

        pred_class = int(model.predict(tfidf_vec)[0])

        # Get probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_vec)[0]
            prob_real = float(probs[0])
            prob_fake = float(probs[1])
        else:
            # Decision function fallback
            df_val = float(model.decision_function(tfidf_vec)[0])
            prob_fake = 1.0 / (1.0 + float(np.exp(-df_val)))
            prob_real = 1.0 - prob_fake

        confidence = prob_fake if pred_class == 1 else prob_real

        # Top keywords / n-grams
        top_keywords = get_top_features_for_text(text, self.vectorizer, top_n=8)

        return {
            "prediction": "FAKE" if pred_class == 1 else "REAL",
            "label": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "REAL": round(prob_real, 4),
                "FAKE": round(prob_fake, 4)
            },
            "top_keywords": top_keywords,
            "model_used": target_model_name if target_model_name != "Best Model" else f"Best Model ({self.best_model_name})"
        }

    def predict_all_models(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Runs prediction across all available models for comparative analysis."""
        results = {}
        for name in ["Logistic Regression", "Support Vector Machine", "Random Forest", "Multinomial Naive Bayes"]:
            if name in self.models:
                results[name] = self.predict(text, model_name=name)
        return results
