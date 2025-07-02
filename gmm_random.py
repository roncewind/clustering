import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


# -----------------------------------------------------------------------------
# visualize the data
def save_component_figure(filename, component_1, component_2):
    plt.scatter(component_1[:, 0], component_1[:, 1], s=0.8)
    plt.scatter(component_2[:, 0], component_2[:, 1], s=0.8)
    plt.title("Gaussian Mixture components")
    plt.axis("equal")
    plt.savefig(filename)
    # plt.show()


# -----------------------------------------------------------------------------
# minimize the bic score
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


# -----------------------------------------------------------------------------

n_samples = 500
np.random.seed(0)
C = np.array([[0.0, -0.1], [1.7, 0.4]])
component_1 = np.dot(np.random.randn(n_samples, 2), C)  # general
component_2 = 0.7 * np.random.randn(n_samples, 2) + np.array([-4, 1])  # spherical

X = np.concatenate([component_1, component_2])

save_component_figure("gaussian_mixture_components.png", component_1, component_2)


# -----------------------------------------------------------------------------
# vary the number of components and the covariance types to find the
# lowest bic score
param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(X)


# -----------------------------------------------------------------------------
# plot the bic scores by creating a data frame with the scores
df = pd.DataFrame(grid_search.cv_results_)[
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
head = df.sort_values(by="BIC score").head(1)
print(head)

# then create an image using seaborn
sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
    hue="Type of covariance",
)
plt.savefig("bic_scores.png")
# plt.show()

# -----------------------------------------------------------------------------
# plot the best model
color_iter = sns.color_palette("tab10", 2)[::-1]
Y_ = grid_search.predict(X)

fig, ax = plt.subplots()

for i, (mean, cov, color) in enumerate(
    zip(
        grid_search.best_estimator_.means_,
        grid_search.best_estimator_.covariances_,
        color_iter,
    )
):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)

plt.title(
    f"Selected GMM: {grid_search.best_params_['covariance_type']} model, "
    f"{grid_search.best_params_['n_components']} components"
)
plt.axis("equal")
plt.savefig("best_model.png")
# plt.show()

# =============================================================================
# create some random clusters
n_samples = 500
# C = np.array([[0.0, -0.1, 0.3], [1.7, 0.4, 2.3], [2.3, 0.2, -1.4]])
# component_1 = np.dot(np.random.randn(n_samples, 3), C)  # general
# component_2 = 0.7 * np.random.randn(n_samples, 3) + np.array([-4, 1, 3])
# component_3 = 0.5 * np.random.randn(n_samples, 3) + np.array([4, 1, -3])
# component_4 = np.random.randn(n_samples, 3)
# X = np.concatenate([component_1, component_2, component_3, component_4])
# print(f'X: {X}')
component_1 = np.random.randn(n_samples, 33)
component_2 = np.random.randn(n_samples, 33) * 0.1
component_3 = np.random.randn(n_samples, 33) * -0.1
component_4 = np.random.randn(n_samples, 33) * 0.5
X = np.concatenate([component_1, component_2, component_3, component_4])

# fit random clusters to a number of components and covariance type:
param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(X)

# find the best fit and output
df = pd.DataFrame(grid_search.cv_results_)[
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
head = df.sort_values(by="BIC score").head(5)
print(head)
