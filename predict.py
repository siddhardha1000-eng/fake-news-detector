"""
Command Line Prediction Tool for Fake News Detection
Usage:
    python predict.py "Your news text here"
    python predict.py --interactive
"""

import sys
import os
import argparse

# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure Windows stdout handles UTF-8 safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.pipeline import FakeNewsPredictor


def display_prediction(res: dict):
    print("\n" + "-" * 60)
    badge = "[FAKE NEWS (SUSPICIOUS)]" if res["prediction"] == "FAKE" else "[AUTHENTIC / REAL NEWS (VERIFIED)]"
    print(f"VERDICT:      {badge}")
    print(f"Confidence:   {res['confidence'] * 100:.2f}%")
    print(f"Model Used:   {res['model_used']}")
    print(f"Probabilities: Real: {res['probabilities']['REAL']*100:.1f}% | Fake: {res['probabilities']['FAKE']*100:.1f}%")
    
    if res["top_keywords"]:
        print("\nTop Contributing Keywords / N-grams:")
        for word, weight in res["top_keywords"]:
            print(f"  • {word:<20} (TF-IDF Weight: {weight:.4f})")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict whether news text is Real or Fake.")
    parser.add_argument("text", nargs="?", default=None, help="News text or headline to classify")
    parser.add_argument("--model", default="Best Model", help="Model name (e.g., 'Logistic Regression', 'Support Vector Machine', 'Random Forest')")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt mode")
    parser.add_argument("--compare", action="store_true", help="Compare predictions across all trained models")

    args = parser.parse_args()

    try:
        predictor = FakeNewsPredictor(models_dir="models_saved")
    except FileNotFoundError:
        print("Model artifacts not found! Automatically training models first...")
        from src.pipeline import PipelineTrainer
        PipelineTrainer().run()
        predictor = FakeNewsPredictor(models_dir="models_saved")

    if args.interactive or not args.text:
        print("\n" + "=" * 60)
        print("   FAKE NEWS DETECTOR - INTERACTIVE COMMAND LINE MODE")
        print("   (Type your headline/article and press Enter. Type 'exit' to quit)")
        print("=" * 60)
        while True:
            try:
                user_text = input("\nEnter News Text: ").strip()
                if user_text.lower() in ["exit", "quit", "q"]:
                    print("Exiting. Goodbye!")
                    break
                if not user_text:
                    continue

                if args.compare:
                    all_res = predictor.predict_all_models(user_text)
                    print("\n--- Multi-Model Comparison ---")
                    for m_name, m_res in all_res.items():
                        print(f"[{m_name}]: {m_res['prediction']} (Confidence: {m_res['confidence']*100:.1f}%)")
                else:
                    res = predictor.predict(user_text, model_name=args.model)
                    display_prediction(res)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    else:
        if args.compare:
            all_res = predictor.predict_all_models(args.text)
            print("\n--- Multi-Model Comparison ---")
            for m_name, m_res in all_res.items():
                print(f"[{m_name}]: {m_res['prediction']} (Confidence: {m_res['confidence']*100:.1f}%)")
        else:
            res = predictor.predict(args.text, model_name=args.model)
            display_prediction(res)


if __name__ == "__main__":
    main()
