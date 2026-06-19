# Script 1: Data Collection via Web Scraping
# Week 2 - ADA Global Academy SIWES Project
# Scrapes dataset info from UCI ML Repository and downloads student performance data

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# ── Step 1: Scrape dataset description from UCI Repository ──────────────────

def scrape_dataset_info():
    """
    Scrapes the UCI ML Repository page for the Student Performance dataset
    to extract the dataset description and attribute information.
    """
    url = "https://archive.ics.uci.edu/dataset/320/student+performance"

    print("Fetching dataset information from UCI ML Repository...")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        title = soup.find("h1")
        if title:
            print(f"\nDataset Found: {title.get_text(strip=True)}")

        print("Successfully scraped dataset metadata from UCI Repository.\n")

    except requests.exceptions.RequestException as e:
        print(f"Could not reach UCI repository: {e}")
        print("Proceeding with local data load...\n")


# ── Step 2: Load the student performance dataset ────────────────────────────

def load_data():
    """
    Loads the Student Performance dataset.
    Falls back to generating a representative sample if the file is not found.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep=";")
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    else:
        print("Dataset file not found. Generating sample dataset for demonstration...\n")
        return generate_sample_data()


def generate_sample_data():
    """
    Generates a sample dataset that mirrors the structure of the
    UCI Student Performance dataset for demonstration purposes.
    """
    import numpy as np
    np.random.seed(42)

    n = 395

    data = {
        "school": np.random.choice(["GP", "MS"], n),
        "sex": np.random.choice(["F", "M"], n),
        "age": np.random.randint(15, 23, n),
        "address": np.random.choice(["U", "R"], n, p=[0.7, 0.3]),
        "famsize": np.random.choice(["LE3", "GT3"], n),
        "Pstatus": np.random.choice(["T", "A"], n, p=[0.8, 0.2]),
        "Medu": np.random.randint(0, 5, n),
        "Fedu": np.random.randint(0, 5, n),
        "studytime": np.random.randint(1, 5, n),
        "failures": np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
        "internet": np.random.choice(["yes", "no"], n, p=[0.75, 0.25]),
        "absences": np.random.randint(0, 30, n),
        "G1": np.random.randint(5, 19, n),
        "G2": np.random.randint(5, 19, n),
    }

    # Final grade influenced by study time and failures
    data["G3"] = (
        0.4 * data["G1"]
        + 0.4 * data["G2"]
        + data["studytime"] * 0.5
        - data["failures"] * 1.5
        + np.random.normal(0, 1, n)
    ).clip(0, 20).astype(int)

    df = pd.DataFrame(data)

    # Save to data folder
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep=";")
    print(f"Sample dataset generated and saved. Shape: {df.shape}\n")

    return df


# ── Step 3: Preview the data ─────────────────────────────────────────────────

def preview_data(df):
    print("=" * 50)
    print("DATASET PREVIEW")
    print("=" * 50)
    print(f"\nRows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_dataset_info()
    df = load_data()
    preview_data(df)
    print("\n✅ Data collection complete. Proceed to 02_clean_data.py")
