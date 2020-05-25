import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist
import pickle
import gc

random_seed = 0
rng = np.random.RandomState(random_seed)  # random_seed

# Data loading
# Iris dataset
n1 = 150  # Number of samples
N1 = 4    # Number of features
i1 = load_iris().data.T
# Normalization
for i in range(N1):
    i1[i] = i1[i]/np.max(i1[i])
i1 = i1.T
ie = pd.read_csv('data/iris_embbeding.csv').values
i2 = pd.read_csv('data/high_dimension_iris.csv').values

# Credit card dataset
n2 = 284807  # Number of samples
N2 = 29    # Number of features
c1 = pd.read_csv('data/creditcard.csv')
# Data reduction
alpha = 1 # percentage of reduction, (0,1)
nr = int(round(n2*alpha,0))
indices = np.random.randint(0,n2,nr)
c1 = c1.drop(indices)
c1 = c1.drop('Time', axis = 1)
c1 = c1.drop('Class', axis = 1)
c1 = c1.values.T
# Normalization
for i in range(N2):
    c1[i] = c1[i]/np.max(c1[i])
c1 = c1.T
ce = pd.read_csv('data/credit_embbeding.csv').values
c2 = pd.read_csv('data/credit_pca.csv').values

# Mountain Clustering
def mountain(X, sigma, metric):
    n,N = X.shape
    gc.enable()
    n_iter = 100

    beta = sigma

    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)

    limits = np.c_[Xmin, Xmax].T

    granularity = 2

    # Create a grid
    xi = []

    for i in range(N):
        xi += [np.linspace(Xmin[i], Xmax[i], granularity)]

    Xi = np.meshgrid(*xi)

    Grid = np.concatenate([
        matrix.flatten().reshape(granularity ** N, 1) for matrix in Xi
    ], axis=1)
    _g = Grid.shape[0]
    # Mountain function
    # a < b indices

    s = 1000
    a = 0
    b = s
    _n = _g
    _m = []

    while _n > 0:
        dist = cdist(Grid[a:b], X, metric)
        m = np.power(dist,2) / (2 * (sigma ** 2))
        m = np.exp(-m).sum(axis=1)
        _m.append(m)
        _n = _n - s
        s = min(_n,s)
        a = b
        b = b + s
    m = np.hstack(_m)
    # Limits
    mmin, mmax = np.minimum(0, m.min()), m.max()
    zlim = [mmin, mmax]
    # Initialize clusters
    C = np.zeros((0, N))
    list_m = []

    # Principal cycle
    i = 1
    while m.max() > zlim[1] / 5 and i < n_iter:
        # Add cluster
        ind = np.argmax(m)
        C = np.r_[C, Grid[ind].reshape(1, N)]

        # Recompute density
        dm = cdist(Grid, C[-1].reshape(1, N), metric)
        dm = np.power(dm,2) / (2 * (beta ** 2))
        dm = m[ind] * np.exp(-dm)

        m = m - dm.flatten()
        list_m += [m.max()]

        # Plot
        i += 1
    return C

# Full size dataset
data = [c1]
sigmas = [0.1,0.5,0.7]
metrics = ['euclidean', 'cosine', 'cityblock']
results_m = {}

for i in range(len(data)):
    X = data[i]
    for metric in metrics:
        for sigma in sigmas:
            results_m[str(i)+str(sigma)+metric] = mountain(X, sigma, metric)

with open('mountain_credit.pickle', 'wb') as handle:
    pickle.dump(results_m, handle, protocol=pickle.HIGHEST_PROTOCOL)
