import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text


warnings.filterwarnings("ignore")



FILE_PATH = Path(__file__).parent / "customers-gp.csv"
OUTPUT_DIR = Path("outputs/clicks")
CHART_DIR = Path("charts/clicks")
MODEL_DIR = Path("models/clicks")
RANDOM_STATE = 42


def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(dataframe):
    dataframe.columns = (
        dataframe.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return dataframe


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    required_cols = [
        "Customer_Name",
        "Username",
        "Points",
        "Birthday",
        "Member_Since",
    ]

    missing_required_cols = [col for col in required_cols if col not in df.columns]
    if missing_required_cols:
        raise ValueError(f"Missing required columns: {missing_required_cols}")

    df["Customer_Name"] = df["Customer_Name"].astype(str).str.strip()
    df["Username"] = df["Username"].astype(str).str.strip()

    df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0)
    df.loc[df["Points"] < 0, "Points"] = 0

    df["Birthday"] = pd.to_datetime(df["Birthday"], errors="coerce", dayfirst=True)
    df["Member_Since"] = pd.to_datetime(
        df["Member_Since"],
        errors="coerce",
        dayfirst=True,
    )

    before_dups = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_dups - len(df)

    return df, duplicates_removed


def save_basic_reports(df):
    missing_report = df.isnull().sum().reset_index()
    missing_report.columns = ["Column", "Missing_Values"]
    missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False)

    summary_stats = pd.DataFrame(
        {
            "Metric": [
                "Customer Count",
                "Mean Points",
                "Median Points",
                "Minimum Points",
                "Maximum Points",
                "Standard Deviation",
                "Q1",
                "Q3",
                "Zero Count",
                "Non-Zero Count",
            ],
            "Value": [
                len(df),
                df["Points"].mean(),
                df["Points"].median(),
                df["Points"].min(),
                df["Points"].max(),
                df["Points"].std(),
                df["Points"].quantile(0.25),
                df["Points"].quantile(0.75),
                (df["Points"] == 0).sum(),
                (df["Points"] > 0).sum(),
            ],
        }
    )
    summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)


def age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age < 20:
        return "Under 20"
    if age < 30:
        return "20-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    return "50+"


def membership_group(days):
    if days < 180:
        return "New Member"
    if days < 365:
        return "Growing Member"
    if days < 730:
        return "Established Member"
    return "Long-Term Member"


def add_engineered_features(df):
    today = pd.Timestamp.today().normalize()

    df["Membership_Days"] = (today - df["Member_Since"]).dt.days
    membership_median = df["Membership_Days"].median()
    df["Membership_Days"] = df["Membership_Days"].fillna(
        0 if pd.isna(membership_median) else membership_median
    )
    df["Membership_Days"] = df["Membership_Days"].clip(lower=0)

    df["Age"] = (today - df["Birthday"]).dt.days / 365.25
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(0 if pd.isna(age_median) else age_median)
    df["Age"] = df["Age"].clip(lower=0)

    df["Log_Points"] = np.log1p(df["Points"])
    df["Sqrt_Points"] = np.sqrt(df["Points"])
    df["Percentile_Rank"] = df["Points"].rank(pct=True) * 100
    df["Points_Per_Day"] = df["Points"] / (df["Membership_Days"] + 1)
    df["Points_Per_Month"] = df["Points"] / ((df["Membership_Days"] / 30) + 1)
    df["Is_Active"] = np.where(df["Points"] > 0, 1, 0)
    df["Age_Group"] = df["Age"].apply(age_group)
    df["Membership_Group"] = df["Membership_Days"].apply(membership_group)

    numeric_features = [
        "Points",
        "Log_Points",
        "Sqrt_Points",
        "Percentile_Rank",
        "Membership_Days",
        "Points_Per_Day",
        "Points_Per_Month",
        "Age",
    ]

    scaled_df = df.copy()
    scaled_df[numeric_features] = MinMaxScaler().fit_transform(df[numeric_features])

    df["Loyalty_Score"] = (
        scaled_df["Points"] * 0.35
        + scaled_df["Log_Points"] * 0.20
        + scaled_df["Percentile_Rank"] * 0.20
        + scaled_df["Points_Per_Day"] * 0.15
        + scaled_df["Membership_Days"] * 0.10
    ) * 100

    return df


def recommendation(segment):
    recommendations = {
        "Most Loyal": "Offer VIP rewards and exclusive loyalty benefits",
        "Highly Active": "Encourage progression with bonus campaigns",
        "Active": "Maintain engagement using regular promotions",
        "Potential": "Target with reminder offers and point boosts",
        "Least Loyal": "Send discount incentives to increase activity",
        "Inactive": "Launch re-engagement campaign",
    }
    return recommendations.get(segment, "Review customer behavior manually")


def add_segments(df):
    q20 = df["Loyalty_Score"].quantile(0.20)
    q40 = df["Loyalty_Score"].quantile(0.40)
    q60 = df["Loyalty_Score"].quantile(0.60)
    q80 = df["Loyalty_Score"].quantile(0.80)

    def assign_segment(row):
        if row["Points"] == 0:
            return "Inactive"
        if row["Loyalty_Score"] >= q80:
            return "Most Loyal"
        if row["Loyalty_Score"] >= q60:
            return "Highly Active"
        if row["Loyalty_Score"] >= q40:
            return "Active"
        if row["Loyalty_Score"] >= q20:
            return "Potential"
        return "Least Loyal"

    df["Customer_Segment"] = df.apply(assign_segment, axis=1)
    df["Business_Recommendation"] = df["Customer_Segment"].apply(recommendation)

    q1 = df["Points"].quantile(0.25)
    q3 = df["Points"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["Outlier_Status"] = np.where(
        (df["Points"] < lower_bound) | (df["Points"] > upper_bound),
        "Outlier",
        "Normal",
    )

    df = df.sort_values(by="Loyalty_Score", ascending=False).reset_index(drop=True)
    df["Loyalty_Rank"] = np.arange(1, len(df) + 1)

    return df


def run_clustering(df):
    cluster_features = df[
        [
            "Points",
            "Log_Points",
            "Membership_Days",
            "Points_Per_Month",
        ]
    ]

    cluster_scaled = MinMaxScaler().fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = kmeans.fit_predict(cluster_scaled)

    cluster_profile = (
        df.groupby("Cluster")
        .agg(
            Customer_Count=("Customer_Name", "count"),
            Avg_Points=("Points", "mean"),
            Avg_Score=("Loyalty_Score", "mean"),
            Avg_Membership_Days=("Membership_Days", "mean"),
        )
        .reset_index()
    )
    cluster_profile.to_csv(OUTPUT_DIR / "cluster_profile.csv", index=False)

    return df


def split_classification_data(df, ml_features):
    X = df[ml_features]
    y = df["Customer_Segment"]

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2 and len(class_counts) > 1

    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y if can_stratify else None,
    )


def slugify_model_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def save_confusion_matrix(name, model, y_test, predictions):
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=model.classes_,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    ax.set_title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{slugify_model_name(name)}_confusion_matrix.png")
    plt.close(fig)


def save_feature_importance(name, model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    slug = slugify_model_name(name)
    importance_df.to_csv(OUTPUT_DIR / f"{slug}_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_title(f"{name} Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{slug}_feature_importance.png")
    plt.close(fig)


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test, feature_names):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1_Score": report["weighted avg"]["f1-score"],
        "Macro_F1_Score": report["macro avg"]["f1-score"],
    }

    slug = slugify_model_name(name)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / f"{slug}_metrics.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"{slug}_classification_report.csv"
    )

    save_confusion_matrix(name, model, y_test, predictions)
    save_feature_importance(name, model, feature_names)

    if name == "Decision Tree":
        tree_rules = export_text(model, feature_names=feature_names)
        with open(MODEL_DIR / "decision_tree_rules.txt", "w", encoding="utf-8") as f:
            f.write("Decision Tree Rules\n")
            f.write(
                "This model classifies customers based on points, activity, "
                "membership duration, and age.\n\n"
            )
            f.write(tree_rules)

    return metrics


def run_classification_models(df):
    ml_features = [
        "Points",
        "Log_Points",
        "Membership_Days",
        "Points_Per_Day",
        "Points_Per_Month",
        "Age",
    ]

    X_train, X_test, y_train, y_test = split_classification_data(df, ml_features)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    model_metrics = []
    for name, model in models.items():
        metrics = evaluate_classifier(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            ml_features,
        )
        model_metrics.append(metrics)

    comparison = pd.DataFrame(model_metrics).sort_values(
        ["F1_Score", "Balanced_Accuracy"],
        ascending=False,
    )
    comparison.to_csv(OUTPUT_DIR / "classification_model_comparison.csv", index=False)

    return comparison


def run_regression_model(df):
    regression_features = df[
        [
            "Membership_Days",
            "Age",
            "Points_Per_Day",
            "Points_Per_Month",
        ]
    ]
    regression_target = df["Loyalty_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        regression_features,
        regression_target,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    predictions = linear_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    regression_metrics = pd.DataFrame(
        {
            "Metric": ["RMSE", "R2 Score"],
            "Value": [rmse, r2],
        }
    )
    regression_metrics.to_csv(OUTPUT_DIR / "linear_regression_metrics.csv", index=False)

    return rmse, r2


def save_customer_outputs(df):
    df.to_csv(OUTPUT_DIR / "segmented_customers.csv", index=False)

    top10 = df.head(10)
    bottom10 = df.sort_values(by="Loyalty_Score").head(10)

    top10.to_csv(OUTPUT_DIR / "top_loyal_customers.csv", index=False)
    bottom10.to_csv(OUTPUT_DIR / "bottom_customers.csv", index=False)

    segment_summary = (
        df.groupby("Customer_Segment")
        .agg(
            Customer_Count=("Customer_Name", "count"),
            Avg_Points=("Points", "mean"),
            Median_Points=("Points", "median"),
            Avg_Loyalty_Score=("Loyalty_Score", "mean"),
        )
        .reset_index()
    )
    segment_summary.to_csv(OUTPUT_DIR / "segment_summary.csv", index=False)

    membership_summary = (
        df.groupby("Membership_Group")
        .agg(
            Customer_Count=("Customer_Name", "count"),
            Avg_Points=("Points", "mean"),
            Avg_Loyalty_Score=("Loyalty_Score", "mean"),
        )
        .reset_index()
    )
    membership_summary.to_csv(
        OUTPUT_DIR / "membership_group_summary.csv",
        index=False,
    )


def save_charts(df):
    top10 = df.head(10)

    plt.figure(figsize=(10, 6))
    plt.hist(df["Points"], bins=20)
    plt.title("Customer Points Distribution")
    plt.xlabel("Points")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "histogram_points.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    df["Customer_Segment"].value_counts().plot(kind="bar")
    plt.title("Customer Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "segment_bar_chart.png")
    plt.close()

    plt.figure(figsize=(8, 8))
    df["Customer_Segment"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.title("Customer Segment Proportion")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "segment_pie_chart.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df["Points"], df["Loyalty_Score"])
    plt.title("Points vs Loyalty Score")
    plt.xlabel("Points")
    plt.ylabel("Loyalty Score")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "points_vs_loyalty_score.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df["Membership_Days"], df["Points"])
    plt.title("Membership Duration vs Points")
    plt.xlabel("Membership Days")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "membership_vs_points.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(top10["Customer_Name"], top10["Loyalty_Score"])
    plt.title("Top 10 Loyal Customers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "top10_loyal_customers.png")
    plt.close()


def main():
    ensure_directories()

    print("CLICKS LOYALTY CUSTOMER ANALYTICS SYSTEM - STANDALONE")


    df, duplicates_removed = load_and_clean_data(FILE_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows after duplicate removal: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Column names: {df.columns.tolist()}")

    save_basic_reports(df)
    df = add_engineered_features(df)
    df = add_segments(df)
    df = run_clustering(df)

    comparison = run_classification_models(df)
    rmse, r2 = run_regression_model(df)

    save_customer_outputs(df)
    save_charts(df)

    print("\nClassification Model Comparison:")
    print(comparison.to_string(index=False))

    print("\nLinear Regression Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Charts saved to: {CHART_DIR}")
    print(f"Model notes saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
