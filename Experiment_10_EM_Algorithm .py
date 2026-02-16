import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Load Data
print("Loading Iris Data for EM Algorithm...")
# We use Iris data for clustering demo (ignoring labels)
try:
    df = pd.read_csv('iris.csv')
    X = df.iloc[:, :-1].values
except FileNotFoundError:
    print("iris.csv not found, using sklearn load_iris")
    iris = load_iris()
    X = iris.data

# EM Algorithm (GMM)
print("\nTraining Gaussian Mixture Model (EM)...")
# Assume 3 components (for 3 iris species)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predict clusters
labels = gmm.predict(X)

print("\nCluster Labels predicted by GMM:")
print(labels)

print("\nModel Converged:", gmm.converged_)
print("Number of Iterations:", gmm.n_iter_)
print("Means of the clusters:\n", gmm.means_)
