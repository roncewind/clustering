# python gmm_analysis.py --csv "<path_to_your_csv_file>.csv" --label_col label --max_components 20 --output_dir "<path_to_output_directory>"
# python gmm_analysis.py --csv "data/all_features.csv" --label_col label --max_components 33 --output_dir "data"

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# =============================================================================
#
def format_seconds_to_hhmmss(seconds):
    """Convert seconds to a formatted string of HH:MM:SS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


# =============================================================================
#
def get_column_labels(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: the class label file '{file_path}' does not exist.")
        exit(1)
    with open(file_path, "r") as file:
        lines = file.readlines()
    print(f"Number of class labels: {len(lines)}")
    column_labels = []
    for line in lines:
        lang = line.strip()
        column_labels.append(lang + "_glyph_count")
        column_labels.append(lang + "_glyph_border")
        column_labels.append(lang + "_prop")
    print(f"Feature columns: {column_labels}")
    return column_labels


# =============================================================================
#
def load_csv_file(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: the feature CSV file '{csv_path}' does not exist.")
        exit(1)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from '{csv_path}'")
    return df


# =============================================================================
#
def run_gmm_analysis(
    df, feature_cols, label_col, min_components, max_components, output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    print("Running GMM analysis...")
    X = df[feature_cols].values
    y_true = df[label_col].values

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize variables
    print("Initializing GMM parameters...")
    print(f"Number of features: {X_scaled.shape[1]}")
    print(f"Number of samples: {X_scaled.shape[0]}")
    print(f"Components range to try: {min_components} - {max_components}")
    lowest_bic = np.inf
    bic_scores = []
    aic_scores = []
    from typing import List, Literal

    n_components_range = range(min_components, max_components + 1)
    cv_types: List[Literal["spherical", "tied", "diag", "full"]] = [
        "spherical",
        "tied",
        "diag",
        "full",
    ]
    best_gmm = None
    best_config = {}

    # Fit models
    fitting_start_time = pd.Timestamp.now()
    for cv_type in cv_types:
        cv_type_time = pd.Timestamp.now()
        elapsed_time = 0
        print(f"Fitting GMM with covariance type: {cv_type} at {cv_type_time}")
        for n_components in n_components_range:
            start_comp_time = pd.Timestamp.now()
            print(
                f"  Number of components: {n_components}/{max_components}: Previous component time: {format_seconds_to_hhmmss(elapsed_time)}, total time: {format_seconds_to_hhmmss((pd.Timestamp.now() - fitting_start_time).total_seconds())}",
                end="\r",
            )
            # if elapsed_time > 0:
            #     print(
            #         f"  Number of components: {n_components}/{max_components}: Previous component time: {format_seconds_to_hhmmss(elapsed_time)}, total time: {(pd.Timestamp.now() - fitting_start_time).total_seconds()}",
            #         end="\r",
            #     )
            # else:
            #     print(
            #         f"  Number of components: {n_components}/{max_components} ",
            #         end="\r",
            #     )
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                random_state=42,
                n_init=2,
            )
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)
            aic = gmm.aic(X_scaled)
            bic_scores.append((cv_type, n_components, bic))
            aic_scores.append((cv_type, n_components, aic))
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                best_config = {
                    "covariance_type": cv_type,
                    "n_components": n_components,
                    "bic": bic,
                    "aic": aic,
                }
            elapsed_time = (pd.Timestamp.now() - start_comp_time).total_seconds()
        print("\n ...fitting complete.")

    # Predict using the best model
    if best_gmm is None:
        print(
            "❌  No valid GMM model was found. Please check your input data and parameters."
        )
        return

    y_pred = best_gmm.predict(X_scaled)

    # Compute clustering metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Save scores
    bic_df = pd.DataFrame(bic_scores, columns=["Covariance Type", "Components", "BIC"])
    aic_df = pd.DataFrame(aic_scores, columns=["Covariance Type", "Components", "AIC"])

    print("📝 Saving BIC and AIC scores.")
    bic_df.to_csv(os.path.join(output_dir, "bic_scores.csv"), index=False)
    aic_df.to_csv(os.path.join(output_dir, "aic_scores.csv"), index=False)

    # Save plots
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=bic_df, x="Components", y="BIC", hue="Covariance Type", marker="o"
    )
    plt.title("BIC Scores by Covariance Type and Number of Components")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bic_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=aic_df, x="Components", y="AIC", hue="Covariance Type", marker="o"
    )
    plt.title("AIC Scores by Covariance Type and Number of Components")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aic_plot.png"))
    plt.close()
    print("📊 BIC and AIC plots saved.")

    # Save best config summary
    print("📝 Saving best GMM configuration and clustering results...")
    with open(os.path.join(output_dir, "best_config.txt"), "w") as f:
        f.write("Best GMM Configuration:\n")
        for k, v in best_config.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nAdjusted Rand Index (ARI): {ari:.4f}\n")
        f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n")

    # Save original data with cluster assignments
    print("📝 Saving clustered data with predictions...")
    df["predicted_cluster"] = y_pred
    df.to_csv(os.path.join(output_dir, "clustered_data.csv"), index=False)

    print("✅ Analysis complete. Results saved to:", output_dir)


# =============================================================================
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GMM clustering analysis")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--label_col", required=True, help="Name of the label column")
    parser.add_argument(
        "--label_file",
        required=False,
        default="data/class_labels.txt",
        help="Path to the label file, default is 'data/class_labels.txt'",
    )
    parser.add_argument(
        "--min_components",
        type=int,
        help="Minimum number of GMM components to try, defaults to 1/3 of feature columns",
    )
    parser.add_argument(
        "--max_components",
        type=int,
        default=20,
        help="Maximum number of GMM components to try",
    )
    parser.add_argument(
        "--output_dir", default="gmm_output", help="Directory to save output results"
    )

    start_time = pd.Timestamp.now()
    print(f"Starting GMM analysis at {start_time}")
    args = parser.parse_args()
    feature_columns = get_column_labels(args.label_file)
    min_components = len(feature_columns) // 3
    if args.min_components is not None:
        min_components = args.min_components

    df = load_csv_file(args.csv)
    # Run the analysis
    run_gmm_analysis(
        df,
        feature_columns,
        args.label_col,
        min_components,
        args.max_components,
        args.output_dir,
    )
    print(f"Finished GMM analysis at {pd.Timestamp.now()}")
    print(
        f"Total time taken: {format_seconds_to_hhmmss((pd.Timestamp.now() - start_time).total_seconds())} seconds"
    )
