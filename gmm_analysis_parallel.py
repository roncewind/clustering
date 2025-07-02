# python gmm_analysis_parallel.py --csv "data/all_features.csv" --label_col label --min_components 11 --max_components 12 --output_dir "data" --n_jobs 8
import argparse
import os
from contextlib import contextmanager

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================================================================
# Context manager for joblib + tqdm
@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


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
def fit_gmm(X_scaled, y_true, cv_type, n_components):
    # print(f"Fitting GMM with {n_components} components and covariance type '{cv_type}'")
    gmm = GaussianMixture(
        n_components=n_components, covariance_type=cv_type, random_state=42, n_init=2
    )
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    y_pred = gmm.predict(X_scaled)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return {
        "cov_type": cv_type,
        "n_components": n_components,
        "bic": bic,
        "aic": aic,
        "ari": ari,
        "nmi": nmi,
        "model": gmm,
    }


# =============================================================================
#
def run_parallel_gmm_analysis(
    csv_path,
    feature_cols,
    label_col,
    min_components,
    max_components,
    output_dir,
    n_jobs,
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_csv_file(csv_path)

    X = df[feature_cols].values
    y_true = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = [
        (cv, n)
        for cv in ["spherical", "tied", "diag", "full"]
        for n in range(min_components, max_components + 1)
    ]

    with tqdm_joblib(tqdm(total=len(param_grid), desc="Fitting GMMs")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_gmm)(X_scaled, y_true, cv_type, n_components)
            for cv_type, n_components in param_grid
        )

    # Collect scores and find best
    scores_df = pd.DataFrame(
        [
            {
                "Covariance Type": r["cov_type"],
                "Components": r["n_components"],
                "BIC": r["bic"],
                "AIC": r["aic"],
                "ARI": r["ari"],
                "NMI": r["nmi"],
            }
            for r in results
        ]
    )
    scores_df.to_csv(os.path.join(output_dir, "gmm_scores.csv"), index=False)

    # Save plots
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=scores_df, x="Components", y="BIC", hue="Covariance Type", marker="o"
    )
    plt.title("BIC Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bic_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=scores_df, x="Components", y="AIC", hue="Covariance Type", marker="o"
    )
    plt.title("AIC Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aic_plot.png"))
    plt.close()

    # Find best model by BIC
    best = min(results, key=lambda r: r["bic"])

    # Save model config
    with open(os.path.join(output_dir, "best_config.txt"), "w") as f:
        f.write(f"Best Model by BIC:\n")
        for k in ["cov_type", "n_components", "bic", "aic", "ari", "nmi"]:
            f.write(f"{k}: {best[k]}\n")

    # Save clustered data
    df["predicted_cluster"] = best["model"].predict(X_scaled)
    df.to_csv(os.path.join(output_dir, "clustered_data.csv"), index=False)

    print("✅ Parallel GMM analysis complete. Results saved to:", output_dir)


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel GMM clustering analysis")
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
        "--max_components", type=int, default=20, help="Max GMM components to test"
    )
    parser.add_argument("--output_dir", default="gmm_output", help="Output directory")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all cores)",
    )

    start_time = pd.Timestamp.now()
    print(f"Starting GMM analysis at {start_time}")
    args = parser.parse_args()
    feature_columns = get_column_labels(args.label_file)
    min_components = len(feature_columns) // 3
    if args.min_components is not None:
        min_components = args.min_components

    run_parallel_gmm_analysis(
        csv_path=args.csv,
        feature_cols=feature_columns,
        label_col=args.label_col,
        min_components=min_components,
        max_components=args.max_components,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(f"Finished GMM analysis at {pd.Timestamp.now()}")
    print(
        f"Total time taken: {format_seconds_to_hhmmss((pd.Timestamp.now() - start_time).total_seconds())} seconds"
    )
