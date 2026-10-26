"""
Main Training Script for Fake News Detection System
Usage:
    python train.py
"""

import sys
import os

# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import PipelineTrainer


def main():
    print("=" * 70)
    print("      FAKE NEWS DETECTION SYSTEM: MACHINE LEARNING PIPELINE")
    print("=" * 70)
    
    trainer = PipelineTrainer(
        dataset_path="data/news_dataset.csv",
        models_dir="models_saved",
        test_size=0.2,
        random_state=42
    )

    trainer.run()
    print("\nModels trained, evaluated, and saved to 'models_saved/' successfully.")
    print("You can now test predictions with:")
    print("    python predict.py \"Your news article headline or text here\"")
    print("Or launch the web interface with:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
