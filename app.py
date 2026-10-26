"""
Streamlit Web Interface for Fake News Detection System
Features:
- Live News Article Classification (Text & URL Scraping)
- Multi-Model Selection & Consensus (Logistic Regression, SVM, Random Forest)
- Prediction Explainability (Top TF-IDF N-grams)
- Model Performance Comparison & Evaluation Dashboard (Precision, Recall, F1, Confusion Matrix)
- Interactive NLP Preprocessing Laboratory
- Real-time News Web Scraper
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Configure page
st.set_page_config(
    page_title="TruthScan - Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure app directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import FakeNewsPredictor, PipelineTrainer
from src.preprocessing import clean_text, tokenize_and_lemmatize, preprocess_text, get_top_features_for_text
from data.scraper import NewsScraper


# Pre-defined test samples for easy testing
SAMPLE_ARTICLES = {
    "Select a pre-loaded sample...": "",
    "Real: NASA Space Telescope Exoplanet Discovery": (
        "BALTIMORE (Scientific American) - Astronomers utilizing high-resolution transmission spectroscopy "
        "from the James Webb Space Telescope have confirmed clear molecular signatures of carbon dioxide "
        "in the atmosphere of exoplanet WASP-39b. The gas giant, orbiting a star 700 light-years away, "
        "provides essential data on chemical evolution and planet formation outside the Solar System. "
        "Findings were published in Nature by the international research consortium."
    ),
    "Real: Federal Reserve Interest Rates Decision": (
        "NEW YORK (Bloomberg) - The Federal Reserve opted to maintain its benchmark interest rate target "
        "range between 5.25% and 5.50% following its two-day policy summit. Federal Reserve Chair Jerome Powell "
        "stated during the press conference that while inflation has decelerated significantly from its peak, "
        "policymakers require continued empirical evidence before initiating rate reductions. Financial markets "
        "responded favorably to the central bank's measured policy stance."
    ),
    "Fake: Secret Pentagon Weather Control Weapon": (
        "SHOCKING: A brave whistleblower from deep inside the Pentagon has just leaked explosive top secret "
        "documents proving what truth-seekers have suspected for decades! Hidden underground military facilities "
        "have been using high frequency electromagnetic waves to trigger earthquakes and hurricanes on command "
        "to manipulate foreign elections. Mainstream media refuses to cover this absolute bombshell revelation! "
        "Share this before the global elites delete it from the internet permanently!"
    ),
    "Fake: Dandelion Water Cancer Miracle Cure Hoax": (
        "MIRACLE CURE: Big Pharma is scrambling to hide this simple everyday kitchen recipe! An 82-year-old "
        "grandmother from Ohio has completely cured over 500 stage 4 cancer patients in just three days using "
        "nothing but boiled dandelion roots and two tablespoons of ordinary baking soda! Greedy pharmaceutical "
        "executives and corrupt oncologists tried to sue her because this miracle 2-dollar home cure destroys "
        "their multi-billion dollar chemotherapy monopoly overnight. Read the secret formula before it gets taken down!"
    )
}


@st.cache_resource
def load_predictor():
    """Caches predictor and ensures models exist."""
    models_dir = os.path.join(BASE_DIR, "models_saved")
    if not os.path.exists(os.path.join(models_dir, "tfidf_vectorizer.joblib")):
        with st.spinner("Initializing and training baseline ML models for the first time..."):
            trainer = PipelineTrainer(
                dataset_path=os.path.join(BASE_DIR, "data", "news_dataset.csv"),
                models_dir=models_dir
            )
            trainer.run()
    return FakeNewsPredictor(models_dir=models_dir)


@st.cache_data
def load_metrics_and_dataset():
    """Loads metrics JSON and dataset preview."""
    metrics_path = os.path.join(BASE_DIR, "models_saved", "model_metrics.json")
    dataset_path = os.path.join(BASE_DIR, "data", "news_dataset.csv")

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    df = pd.DataFrame()
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)

    return metrics, df


def main():
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 25px 0;">
            <h1 style="color: #1E3A8A; margin-bottom: 4px;">📰 TruthScan: Fake News Detection System</h1>
            <p style="font-size: 1.15rem; color: #4B5563;">
                Machine Learning-Powered Authenticity Verification using <b>TF-IDF N-grams</b> & <b>Supervised Classifiers</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize predictor
    try:
        predictor = load_predictor()
        metrics, dataset_df = load_metrics_and_dataset()
    except Exception as e:
        st.error(f"Error loading system: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/news.png", width=70)
        st.header("⚙️ Configuration")
        
        selected_model = st.selectbox(
            "Select Machine Learning Model",
            options=["Best Model", "Logistic Regression", "Support Vector Machine", "Random Forest", "Multinomial Naive Bayes"],
            index=0,
            help="Choose the classifier to perform inference."
        )

        st.markdown("---")
        st.subheader("📊 System Specs")
        if metrics:
            st.write(f"**Best Model:** `{metrics.get('best_model_name', 'N/A')}`")
            st.write(f"**Vocab Size (N-grams):** `{metrics.get('vocab_size', 5000):,}`")
            st.write(f"**Training Corpus:** `{metrics.get('total_samples', 0):,} articles`")
            st.write(f"**Test Set:** `{metrics.get('test_samples', 0):,} articles`")

        st.markdown("---")
        if st.button("🔄 Retrain All Models"):
            with st.spinner("Retraining all models on dataset..."):
                trainer = PipelineTrainer(
                    dataset_path=os.path.join(BASE_DIR, "data", "news_dataset.csv"),
                    models_dir=os.path.join(BASE_DIR, "models_saved")
                )
                trainer.run()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Retraining complete!")
                st.rerun()

        st.markdown("""
        <div style="font-size: 0.85rem; color: #6B7280; margin-top: 20px;">
            <b>Antigravity ML Pipeline</b><br>
            • Feature Extraction: TF-IDF (1, 2) n-grams<br>
            • Classifiers: LR, SVM, Random Forest<br>
            • Evaluation: Precision, Recall, F1
        </div>
        """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 News Verification",
        "📈 Model Comparison & Metrics",
        "🌐 Live Web Scraper",
        "🧪 NLP Preprocessing Lab"
    ])

    # ==========================================
    # TAB 1: NEWS VERIFICATION
    # ==========================================
    with tab1:
        st.subheader("Analyze News Content for Authenticity")
        
        col_sample, col_spacer = st.columns([2, 1])
        with col_sample:
            sample_choice = st.selectbox(
                "Load an example article to test immediately:",
                options=list(SAMPLE_ARTICLES.keys()),
                index=0
            )

        # Default text based on sample choice
        default_val = SAMPLE_ARTICLES[sample_choice] if sample_choice in SAMPLE_ARTICLES else ""

        user_input = st.text_area(
            "Paste the news headline or article body below:",
            value=default_val,
            height=160,
            placeholder="e.g. Breaking: Officials confirmed new infrastructure bill passes senate..."
        )

        col_btn, col_chk = st.columns([1, 2])
        with col_btn:
            check_button = st.button("🔎 Verify Authenticity", type="primary", use_container_width=True)
        with col_chk:
            show_all_models = st.checkbox("Compare predictions across all 4 models simultaneously", value=True)

        if check_button:
            if not user_input.strip():
                st.warning("Please enter or paste news text before verifying.")
            else:
                with st.spinner("Analyzing linguistic patterns and n-gram representations..."):
                    res = predictor.predict(user_input, model_name=selected_model)

                st.markdown("---")
                
                # Main Verdict Card
                is_fake = (res["prediction"] == "FAKE")
                verdict_color = "#DC2626" if is_fake else "#16A34A"
                verdict_bg = "#FEF2F2" if is_fake else "#F0FDF4"
                verdict_border = "#F87171" if is_fake else "#86EFAC"
                verdict_icon = "⚠️" if is_fake else "✅"
                verdict_text = "POTENTIALLY FAKE / FABRICATED NEWS" if is_fake else "AUTHENTIC / CREDIBLE NEWS"

                st.markdown(f"""
                <div style="background-color: {verdict_bg}; border: 2px solid {verdict_border}; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <h2 style="color: {verdict_color}; margin: 0; font-size: 1.6rem;">{verdict_icon} {verdict_text}</h2>
                            <p style="color: #374151; margin-top: 5px; margin-bottom: 0; font-size: 1rem;">
                                Evaluated using: <b>{res['model_used']}</b>
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 2rem; font-weight: bold; color: {verdict_color};">{res['confidence'] * 100:.1f}%</span><br>
                            <span style="color: #6B7280; font-size: 0.9rem;">Confidence</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities & Details
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.write("**Credibility Distribution:**")
                    st.progress(res["probabilities"]["REAL"], text=f"Authentic (Real): {res['probabilities']['REAL']*100:.1f}%")
                    st.progress(res["probabilities"]["FAKE"], text=f"Fabricated (Fake): {res['probabilities']['FAKE']*100:.1f}%")

                with col_prob2:
                    st.write("**Linguistic Explainability (Top Indicative N-Grams):**")
                    if res["top_keywords"]:
                        kw_df = pd.DataFrame(res["top_keywords"], columns=["N-gram Feature", "TF-IDF Weight"])
                        st.dataframe(kw_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No strong vocabulary n-grams matched the training lexicon.")

                # Multi-model consensus comparison
                if show_all_models:
                    st.markdown("#### 🤖 Multi-Model Consensus Breakdown")
                    all_preds = predictor.predict_all_models(user_input)
                    cols = st.columns(len(all_preds))
                    
                    for col, (m_name, m_res) in zip(cols, all_preds.items()):
                        m_fake = (m_res["prediction"] == "FAKE")
                        color = "red" if m_fake else "green"
                        icon = "⚠️ Fake" if m_fake else "✅ Real"
                        with col:
                            st.metric(
                                label=m_name,
                                value=icon,
                                delta=f"{m_res['confidence']*100:.1f}% conf",
                                delta_color="normal"
                            )

    # ==========================================
    # TAB 2: MODEL COMPARISON & METRICS
    # ==========================================
    with tab2:
        st.subheader("Model Evaluation & Comparative Benchmark")
        st.write(
            "Evaluation performed on held-out stratified test set using **Precision**, **Recall**, "
            "**F1-Score (Macro & Fake class)**, **Accuracy**, and **ROC-AUC**."
        )

        if metrics and "comparison_summary" in metrics:
            comp_df = pd.DataFrame(metrics["comparison_summary"])
            
            # Format dataframe for display
            st.dataframe(
                comp_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1-Score (Fake)", "F1-Score (Macro)"], color="#DCFCE7"),
                use_container_width=True,
                hide_index=True
            )

            # Bar Chart Comparison
            st.markdown("#### 📊 Metric Comparison Across Classifiers")
            chart_data = comp_df.set_index("Model")[["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1-Score (Macro)"]]
            st.bar_chart(chart_data)

            # Confusion Matrix Viewer
            st.markdown("---")
            st.markdown("#### 🎯 Confusion Matrix & Classification Report")
            model_cm_choice = st.selectbox(
                "Select Model to Inspect Confusion Matrix:",
                options=list(metrics.get("detailed_metrics", {}).keys())
            )

            if model_cm_choice in metrics.get("detailed_metrics", {}):
                m_details = metrics["detailed_metrics"][model_cm_choice]
                
                col_cm1, col_cm2 = st.columns([1, 1])
                with col_cm1:
                    st.write(f"**Confusion Matrix for {model_cm_choice}:**")
                    cm_data = pd.DataFrame(
                        [
                            [m_details["tn"], m_details["fp"]],
                            [m_details["fn"], m_details["tp"]]
                        ],
                        index=["Actual Real (0)", "Actual Fake (1)"],
                        columns=["Pred Real (0)", "Pred Fake (1)"]
                    )
                    st.table(cm_data)
                    st.caption("TN: True Negatives | FP: False Positives | FN: False Negatives | TP: True Positives")

                with col_cm2:
                    st.write(f"**Detailed Classification Report:**")
                    rep = m_details["classification_report"]
                    rep_rows = []
                    for k in ["Real News (0)", "Fake News (1)", "macro avg", "weighted avg"]:
                        if k in rep:
                            rep_rows.append({
                                "Class / Average": k,
                                "Precision": round(rep[k]["precision"], 3),
                                "Recall": round(rep[k]["recall"], 3),
                                "F1-Score": round(rep[k]["f1-score"], 3),
                                "Support": int(rep[k]["support"])
                            })
                    st.dataframe(pd.DataFrame(rep_rows), hide_index=True, use_container_width=True)

            # Training Times
            if "training_times" in metrics:
                st.markdown("#### ⏱️ Training Duration")
                time_cols = st.columns(len(metrics["training_times"]))
                for col, (m_n, t_s) in zip(time_cols, metrics["training_times"].items()):
                    col.metric(m_n, f"{t_s:.3f} s")
        else:
            st.info("Metrics not found. Click 'Retrain All Models' in the sidebar.")

    # ==========================================
    # TAB 3: LIVE WEB SCRAPER
    # ==========================================
    with tab3:
        st.subheader("Scrape & Verify Live News URL")
        st.write("Extract headlines and body paragraphs directly from online news websites.")

        url_input = st.text_input(
            "Enter Article URL:",
            placeholder="https://www.reuters.com/... or https://en.wikipedia.org/wiki/Artificial_intelligence"
        )

        scrape_btn = st.button("🌐 Scrape Article Content", type="primary")

        if scrape_btn:
            if not url_input.strip():
                st.warning("Please provide a valid article URL.")
            else:
                with st.spinner("Scraping and parsing article content..."):
                    scraper = NewsScraper()
                    scraped_data = scraper.scrape_article(url_input)

                if scraped_data.get("error"):
                    st.error(f"Failed to scrape: {scraped_data['error']}")
                else:
                    st.success("Article successfully fetched and extracted!")
                    st.write(f"**Extracted Headline:** {scraped_data['title']}")
                    st.write(f"**Content Length:** {len(scraped_data['text'])} characters")
                    
                    with st.expander("Preview Scraped Article Body", expanded=False):
                        st.write(scraped_data["text"][:1500] + ("..." if len(scraped_data["text"]) > 1500 else ""))

                    # Direct Prediction on Scraped Text
                    st.markdown("---")
                    st.subheader("Instant Authenticity Check on Scraped Article")
                    with st.spinner("Predicting..."):
                        p_res = predictor.predict(scraped_data["full_text"], model_name=selected_model)
                    
                    p_fake = (p_res["prediction"] == "FAKE")
                    color = "red" if p_fake else "green"
                    badge = "⚠️ FAKE NEWS" if p_fake else "✅ AUTHENTIC NEWS"
                    st.markdown(f"### Result: :{color}[{badge}] (Confidence: {p_res['confidence']*100:.1f}%)")
                    st.write(f"Model used: `{p_res['model_used']}`")

    # ==========================================
    # TAB 4: NLP PREPROCESSING LAB
    # ==========================================
    with tab4:
        st.subheader("Interactive NLP Preprocessing Pipeline Explorer")
        st.write("Observe each stage of text preprocessing: normalization, tokenization, stopword removal, lemmatization, and TF-IDF n-grams.")

        lab_text = st.text_area(
            "Enter sample text to inspect transformation steps:",
            value="BREAKING NEWS: The NASA scientists haven't found aliens living on the Moon yet! Check out https://example.com for more info.",
            height=100
        )

        if st.button("⚡ Run Preprocessing Steps"):
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("**Step 1: Raw Input**")
                st.code(lab_text)

                cleaned = clean_text(lab_text)
                st.write("**Step 2: Cleaned & Normalized (Lowercased, URLs removed, contractions expanded)**")
                st.code(cleaned)

            with c2:
                tokens = tokenize_and_lemmatize(lab_text)
                st.write(f"**Step 3: Tokenized, Lemmatized & Stopwords Removed ({len(tokens)} tokens)**")
                st.write(tokens)

                final_str = preprocess_text(lab_text)
                st.write("**Step 4: Final Preprocessed String for TF-IDF**")
                st.code(final_str)

            # TF-IDF N-grams representation
            st.markdown("---")
            st.write("**Step 5: TF-IDF N-grams Vectorization Breakdown**")
            if predictor.vectorizer:
                features = get_top_features_for_text(lab_text, predictor.vectorizer, top_n=12)
                if features:
                    feat_df = pd.DataFrame(features, columns=["N-gram Term (Unigram / Bigram)", "TF-IDF Weight"])
                    st.dataframe(feat_df, use_container_width=True)
                else:
                    st.info("None of the terms in this specific text exist in the trained TF-IDF vocabulary.")

        st.markdown("---")
        st.subheader("📚 Dataset Sample Browser")
        if not dataset_df.empty:
            category_filter = st.selectbox(
                "Filter by Label:",
                options=["All", "Authentic / Real (0)", "Fabricated / Fake (1)"]
            )
            filtered_df = dataset_df.copy()
            if category_filter == "Authentic / Real (0)":
                filtered_df = filtered_df[filtered_df["label"] == 0]
            elif category_filter == "Fabricated / Fake (1)":
                filtered_df = filtered_df[filtered_df["label"] == 1]

            st.dataframe(
                filtered_df[["title", "category", "label", "text"]].head(15),
                use_container_width=True,
                height=250
            )
            st.caption(f"Showing sample articles from total dataset of {len(dataset_df)} records.")


if __name__ == "__main__":
    main()
