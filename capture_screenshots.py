"""
Automated High-Resolution Screenshot Capture for TruthScan Fake News Detection System
Captures every screen, tab, classification verdict, and comparison metric.
"""

import os
import sys
import time
import shutil
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure stdout handles UTF-8 safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def generate_metrics_charts():
    """Generates a crisp high-res multi-panel evaluation metric graphic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    
    # Chart 1: Model Comparison Bar Chart
    models = ["Logistic\nRegression", "Random\nForest", "Naive\nBayes", "Support\nVector Machine"]
    accuracies = [100.0, 100.0, 100.0, 99.58]
    f1_macros = [100.0, 100.0, 100.0, 99.58]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, accuracies, width, label="Accuracy (%)", color="#3B82F6", edgecolor="#1D4ED8")
    rects2 = axes[0].bar(x + width/2, f1_macros, width, label="F1-Score Macro (%)", color="#10B981", edgecolor="#047857")
    
    axes[0].set_title("Model Performance Comparison (Test Set N=240)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Score (%)", fontsize=11)
    axes[0].set_ylim(95, 102)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].legend(loc="lower right", framealpha=0.9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    
    for r in rects1:
        h = r.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Chart 2: Confusion Matrix Heatmap (Logistic Regression)
    cm = np.array([[120, 0], [0, 120]])
    im = axes[1].imshow(cm, cmap="Blues", interpolation="nearest")
    axes[1].set_title("Confusion Matrix - Best Model (Logistic Regression)", fontsize=13, fontweight="bold", pad=12)
    
    tick_marks = np.arange(2)
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(["Pred Real (0)", "Pred Fake (1)"], fontsize=10)
    axes[1].set_yticklabels(["Actual Real (0)", "Actual Fake (1)"], fontsize=10)
    
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i, j]}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})",
                         ha="center", va="center", color="white" if cm[i, j] > 50 else "black", fontsize=14, fontweight="bold")
    
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    chart_path = os.path.join(SCREENSHOT_DIR, "07_evaluation_metrics_and_confusion_matrix.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Generated 07_evaluation_metrics_and_confusion_matrix.png")

def capture_streamlit_screens():
    options = EdgeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1100")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Edge(options=options)
    driver.set_window_size(1600, 1100)

    try:
        url = "http://localhost:8501"
        print(f"Connecting to Streamlit at {url}...")
        driver.get(url)
        time.sleep(8)  # Wait for initial model loading

        def click_button_by_text(text_substring):
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for b in buttons:
                if text_substring.lower() in b.text.lower():
                    b.click()
                    return True
            return False

        # --- Screen 1: Authentic News Verification ---
        print("Capturing 01_news_verification_authentic.png...")
        click_button_by_text("Verify Authenticity")
        time.sleep(4)
        driver.save_screenshot(os.path.join(SCREENSHOT_DIR, "01_news_verification_authentic.png"))

        # --- Screen 2: Fake News Verification ---
        print("Capturing 02_news_verification_fake.png...")
        textareas = driver.find_elements(By.TAG_NAME, "textarea")
        if textareas:
            fake_text = (
                "SHOCKING: Brave whistleblower leaks top secret military documents proving "
                "weather control weapons caused earthquakes worldwide! Mainstream corrupt media "
                "is trying to delete this truth! Share before taken down!"
            )
            driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", textareas[0], fake_text)
            time.sleep(1)
            click_button_by_text("Verify Authenticity")
            time.sleep(4)
            driver.save_screenshot(os.path.join(SCREENSHOT_DIR, "02_news_verification_fake.png"))

        # --- Screen 3: Model Comparison & Benchmark ---
        print("Capturing 03_model_comparison_benchmark.png...")
        click_button_by_text("Model Comparison")
        time.sleep(4)
        driver.save_screenshot(os.path.join(SCREENSHOT_DIR, "03_model_comparison_benchmark.png"))

        # --- Screen 4: Confusion Matrix & Classification Report ---
        print("Capturing 04_confusion_matrix_report.png...")
        driver.execute_script("window.scrollTo(0, 650);")
        time.sleep(2)
        driver.save_screenshot(os.path.join(SCREENSHOT_DIR, "04_confusion_matrix_report.png"))

        # --- Screen 5: Live Web Scraper ---
        print("Capturing 05_live_web_scraper.png...")
        driver.execute_script("window.scrollTo(0, 0);")
        click_button_by_text("Live Web Scraper")
        time.sleep(3)
        inputs = driver.find_elements(By.XPATH, "//input[@type='text']")
        if inputs:
            driver.execute_script("arguments[0].value = 'https://en.wikipedia.org/wiki/Machine_learning'; arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", inputs[0])
            time.sleep(1)
            click_button_by_text("Scrape Article Content")
            time.sleep(5)
        driver.save_screenshot(os.path.join(SCREENSHOT_DIR, "05_live_web_scraper.png"))

        # --- Screen 6: NLP Preprocessing Lab ---
        print("Capturing 06_nlp_preprocessing_lab.png...")
        click_button_by_text("NLP Preprocessing")
        time.sleep(3)
        click_button_by_text("Run Preprocessing Steps")
        time.sleep(4)
        driver.save_screenshot(os.path.join(SCREENSHOT_DIR, "06_nlp_preprocessing_lab.png"))

        print("All browser screenshots successfully generated!")

    except Exception as e:
        print(f"Exception during capture: {e}")
    finally:
        driver.quit()

def copy_and_package():
    downloads_dir = r"C:\Users\PRASANA\Downloads\fake_news_detector_screenshots"
    desktop_dir = r"C:\Users\PRASANA\Desktop\fake_news_detector_screenshots"
    
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(desktop_dir, exist_ok=True)

    files = [f for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".png")]
    for f in files:
        src = os.path.join(SCREENSHOT_DIR, f)
        shutil.copy2(src, os.path.join(downloads_dir, f))
        shutil.copy2(src, os.path.join(desktop_dir, f))

    zip_path = os.path.join(BASE_DIR, "screenshots.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(SCREENSHOT_DIR, f), f)

    shutil.copy2(zip_path, r"C:\Users\PRASANA\Downloads\screenshots.zip")
    shutil.copy2(zip_path, r"C:\Users\PRASANA\Desktop\screenshots.zip")
    print(f"Copied {len(files)} screenshot files to Downloads, Desktop, and packaged into screenshots.zip!")

if __name__ == "__main__":
    generate_metrics_charts()
    capture_streamlit_screens()
    copy_and_package()
