# Tinker about with clustering


## metrics to use when evaluating clustering algorithms:

| Metric  | Use For	           | Range	     | Interpretation                        |
| ------- | ------------------ | :---------: | ------------------------------------- |
| BIC     | Model selection    |  −∞ to +∞   | Lower is better                       |
| AIC     | Model selection    |  −∞ to +∞   | Lower is better                       |
| ARI     | Clustering quality |	−1 to 1  | 1 = perfect match, 0 = random         |
| NMI     | Clustering quality |	0 to 1   | 1 = perfect overlap, 0 = independent  |


## Set up virtual environment for Python

```console
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip freeze > requirements.txt
```

## Example:

https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py


## TODO with gmm_analysis_parallel:

- Evaluating or visualizing the clustering results
- Switching to a GPU implementation with cuML
- Logging or resuming long jobs, save long jobs with `pickle`
- Automating feature selection or PCA tuning
