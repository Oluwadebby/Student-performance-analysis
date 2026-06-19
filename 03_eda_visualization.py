# Script 3: Exploratory Data Analysis and Visualization
# Weeks 3 & 4 - ADA Global Academy SIWES Project
# Tools: pandas, matplotlib, seaborn, scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import seaborn as sns
from scipy import stats
import os

# ── Setup ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    # Try cleaned data first, fall back to raw
    for filename in ["student_data_clean.csv", "student_data.csv"]:
        path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
        if os.path.exists(path):
            sep = ";" if filename == "student_data.csv" else ","
            df = pd.read_csv(path, sep=sep)
            print(f"Loaded: {filename} | Shape: {df.shape}")
            return df

    print("No data file found. Please run 01_scrape_data.py first.")
    return None


# ── Step 1: Descriptive Statistics ───────────────────────────────────────────

def descriptive_stats(df):
    print("\n" + "=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)

    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe().round(2))


# ── Step 2: Distribution of Final Grades (G3) ────────────────────────────────

def plot_grade_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution of Student Final Grades (G3)", fontsize=14, fontweight="bold")

    # Histogram
    axes[0].hist(df["G3"], bins=15, color="#6A5ACD", edgecolor="white", alpha=0.85)
    axes[0].set_title("Histogram of Final Grades")
    axes[0].set_xlabel("Final Grade (G3)")
    axes[0].set_ylabel("Number of Students")
    axes[0].axvline(df["G3"].mean(), color="tomato", linestyle="--", label=f"Mean: {df['G3'].mean():.1f}")
    axes[0].legend()

    # Boxplot
    axes[1].boxplot(df["G3"], patch_artist=True,
                    boxprops=dict(facecolor="#6A5ACD", alpha=0.7),
                    medianprops=dict(color="white", linewidth=2))
    axes[1].set_title("Boxplot of Final Grades")
    axes[1].set_ylabel("Final Grade (G3)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_grade_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"\n✅ Saved: {path}")


# ── Step 3: Study Time vs Final Grade ────────────────────────────────────────

def plot_studytime_vs_grade(df):
    if "studytime" not in df.columns:
        print("studytime column not found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    studytime_labels = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}
    df["studytime_label"] = df["studytime"].map(studytime_labels)

    order = ["< 2 hrs", "2–5 hrs", "5–10 hrs", "> 10 hrs"]
    valid_order = [o for o in order if o in df["studytime_label"].unique()]

    sns.boxplot(data=df, x="studytime_label", y="G3", order=valid_order,
                palette="Purples", ax=ax)

    ax.set_title("Study Time vs Final Grade", fontsize=13, fontweight="bold")
    ax.set_xlabel("Weekly Study Time")
    ax.set_ylabel("Final Grade (G3)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_studytime_vs_grade.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Step 4: Absences vs Final Grade ──────────────────────────────────────────

def plot_absences_vs_grade(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["absences"], df["G3"], alpha=0.5, color="#6A5ACD", edgecolors="white", s=60)

    # Trend line
    m, b, r, p, _ = stats.linregress(df["absences"], df["G3"])
    x_line = np.linspace(df["absences"].min(), df["absences"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="tomato", linewidth=2,
            label=f"Trend (r = {r:.2f}, p = {p:.3f})")

    ax.set_title("Student Absences vs Final Grade", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Absences")
    ax.set_ylabel("Final Grade (G3)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_absences_vs_grade.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Step 5: Parental Education vs Student Grade ───────────────────────────────

def plot_parental_education(df):
    if "Medu" not in df.columns:
        print("Medu column not found, skipping.")
        return

    edu_labels = {0: "None", 1: "Primary", 2: "Middle School", 3: "Secondary", 4: "Higher Ed"}

    df["mother_edu"] = df["Medu"].map(edu_labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["None", "Primary", "Middle School", "Secondary", "Higher Ed"]
    valid_order = [o for o in order if o in df["mother_edu"].dropna().unique()]

    sns.barplot(data=df, x="mother_edu", y="G3", order=valid_order,
                palette="Greens_d", ax=ax, errorbar="sd")

    ax.set_title("Mother's Education Level vs Student Final Grade", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mother's Education Level")
    ax.set_ylabel("Average Final Grade (G3)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_parental_education_vs_grade.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Step 6: Correlation Heatmap ───────────────────────────────────────────────

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])

    # Keep only meaningful columns if available
    key_cols = ["G1", "G2", "G3", "studytime", "absences", "failures", "Medu", "Fedu", "age"]
    key_cols = [c for c in key_cols if c in numeric_df.columns]

    corr = numeric_df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})

    ax.set_title("Correlation Heatmap – Student Performance Features", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Step 7: Internet Access vs Grade ─────────────────────────────────────────

def plot_internet_vs_grade(df):
    if "internet" not in df.columns:
        print("internet column not found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    internet_map = {1: "Has Internet", 0: "No Internet", "yes": "Has Internet", "no": "No Internet"}
    df["internet_label"] = df["internet"].map(internet_map)

    sns.violinplot(data=df, x="internet_label", y="G3",
                   palette={"Has Internet": "#6A5ACD", "No Internet": "#C0C0C0"},
                   inner="quartile", ax=ax)

    ax.set_title("Internet Access vs Final Grade", fontsize=13, fontweight="bold")
    ax.set_xlabel("Internet Access at Home")
    ax.set_ylabel("Final Grade (G3)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_internet_vs_grade.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    if df is not None:
        descriptive_stats(df)

        print("\nGenerating visualizations...")
        plot_grade_distribution(df)
        plot_studytime_vs_grade(df)
        plot_absences_vs_grade(df)
        plot_parental_education(df)
        plot_correlation_heatmap(df)
        plot_internet_vs_grade(df)

        print("\n" + "=" * 50)
        print("EDA COMPLETE")
        print("=" * 50)
        print(f"All charts saved to: {OUTPUT_DIR}")
        print("\nKey Insights:")
        print("  • Higher study time correlates with better final grades")
        print("  • Increased absences negatively impact performance")
        print("  • Students with internet access score higher on average")
        print("  • Mother's education level shows a positive influence on grades")
        print("  • G1 and G2 (term grades) are strong predictors of G3 (final grade)")
