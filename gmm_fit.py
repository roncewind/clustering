import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

number_of_features = 33

# maximum number of clusters to fit
max_number_of_components = 100


# -----------------------------------------------------------------------------
# minimize the bic score
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


# -----------------------------------------------------------------------------
# create some random clusters
def get_random_samples(n_samples, n_components):
    # C = np.array([[0.0, -0.1, 0.3], [1.7, 0.4, 2.3], [2.3, 0.2, -1.4]])
    # component_1 = np.dot(np.random.randn(n_samples, number_of_features), C)  # general
    # component_2 = 0.7 * np.random.randn(n_samples, number_of_features) + np.array([-4, 1, 3])
    # component_3 = 0.5 * np.random.randn(n_samples, number_of_features) + np.array([4, 1, -3])
    # component_4 = np.random.randn(n_samples, number_of_features)
    # X = np.concatenate([component_1, component_2, component_3, component_4])

    # component_1 = np.random.randn(n_samples, number_of_features)
    # component_2 = np.random.randn(n_samples, number_of_features) * 0.1
    # component_3 = np.random.randn(n_samples, number_of_features) * -0.1
    # component_4 = np.random.randn(n_samples, number_of_features) * 0.5
    # X = np.concatenate([component_1, component_2, component_3, component_4])
    comps = []
    for _ in range(1, n_components + 1):
        comps.append(np.random.randn(n_samples, number_of_features) * np.random.randn())
    X = np.concatenate(comps)
    # print(f'X: {X}')
    # print(f'len X: {len(X)}')
    return X


# -----------------------------------------------------------------------------
# read file with business name data
def get_name_data():
    print("Reading data...", flush=True)
    X = pd.DataFrame()
    file_path = "~/senzing-garage.git/bizname-research/spike/ron/model_build/n_comp12_270k_3/business_cluster_data_abeeijrttuz_curated_pt"
    for n in range(1, 12):
        filename = f'{file_path}{n}.csv'
        df = pd.read_csv(filename)
        df = df.drop(columns=['Unnamed: 0', 'Culture', 'Name'], axis=1)
        X = pd.concat([X, df])
    print(X)
    print("Data read.", flush=True)
    return X.to_numpy()


# -----------------------------------------------------------------------------
# read file with business name data
def get_feature_data():
    print("Reading data...", flush=True)
    X = pd.DataFrame()
    file_path = "~/senzing-garage.git/bizname-research/spike/ron/nn-data/data/all_features.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=['label', 'Name'], axis=1)
    X = pd.concat([X, df])
    print(X)
    print("Data read.", flush=True)
    return X.to_numpy()


# =============================================================================
# create some random clusters
# X = get_random_samples(n_samples=500, n_components=4)
# X = get_name_data()
X = get_feature_data()

# # fit clusters to a number of components and covariance type:
# param_grid = {
#     "n_components": range(11, max_number_of_components),
#     "covariance_type": ["spherical", "tied", "diag", "full"],
# }
# grid_search = GridSearchCV(
#     GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
# )
# print("Starting grid search fit.")
# grid_search.fit(X)

# # find the best fit and output
# df = pd.DataFrame(grid_search.cv_results_)[
#     ["param_n_components", "param_covariance_type", "mean_test_score"]
# ]
# df["mean_test_score"] = -df["mean_test_score"]

# df = df.rename(
#     columns={
#         "param_n_components": "Number of components",
#         "param_covariance_type": "Type of covariance",
#         "mean_test_score": "BIC score",
#     }
# )
# best_fit = df.sort_values(by="BIC score").head(5)
# print("Top 5 best fit:")
# print(best_fit)


param_distributions = {
    'n_components': np.arange(50, 133),
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
}
gmm = GaussianMixture()
random_search = RandomizedSearchCV(
    estimator=gmm,
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings to sample
    cv=5,       # Number of cross-validation folds
    verbose=2,  # Increase verbosity for more detailed output
    random_state=42,  # For reproducible results
    n_jobs=10  # number of CPUs -1 means all
)
print("Random search fit...")
random_search.fit(X)
# best_params = random_search.best_params_
best_gmm_model = random_search.best_estimator_
print("-----------------------------------")
print("Best model:")
print(best_gmm_model)

# find the best fit and output
df = pd.DataFrame(random_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]

df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
best_fit = df.sort_values(by="BIC score").head(5)
print("-----------------------------------")
print("Top 5 best fit:")
print(best_fit)

