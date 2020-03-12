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
ie = pd.read_csv('iris_embbeding.csv').values
i2 = pd.read_csv('high_dimension_iris.csv').values

# Credit card dataset
n2 = 284807  # Number of samples
N2 = 29    # Number of features
c1 = pd.read_csv('creditcard.csv')
c1 = c1.drop('Time', axis = 1)
c1 = c1.drop('Class', axis = 1)
c1 = c1.values.T
# Normalization
for i in range(N2):
    c1[i] = c1[i]/np.max(c1[i])
c1 = c1.T
ce = pd.read_csv('credit_embbeding.csv').values
c2 = pd.read_csv('credit_pca.csv').values

# Subtractive clustering
def subtractive(X,r_a,metric):
    n,N = X.shape
    n_iter = 100
    gc.enable()
    # Parameters
    r_b = 1.5*r_a
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    limits = np.c_[Xmin, Xmax].T
    # Mountain function
    s = 1000
    a = 0
    b = s
    _n = n
    _m = []
    while _n > 0:
        dist = cdist(X[a:b], X, metric)
        m = dist / (2 * ((r_a / 2) ** 2))
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

    # Principal cycle
    i = 1
    while m.max() > zlim[1] / 5 and i < n_iter:
        # Add cluster
        ind = np.argmax(m)
        C = np.r_[C, X[ind].reshape(1, N)]

        # Recompute density
        dm = cdist(X, C[-1].reshape(1, N), metric)
        dm = dm / (2 * ((r_b / 2) ** 2))
        dm = m[ind] * np.exp(-dm)

        m = m - dm.flatten()
        i += 1
    return C

# Mountain Clustering
def mountain(X, sigma, metric):
    n,N = X.shape
    gc.enable()
    n_iter = 100

    beta = sigma

    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)

    limits = np.c_[Xmin, Xmax].T

    granularity = 5

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
        dm = dm / (2 * (beta ** 2))
        dm = m[ind] * np.exp(-dm)

        m = m - dm.flatten()
        list_m += [m.max()]

        # Plot
        i += 1
    return len(C)

def kmeans(X,k,metric):
    epsilon = 0.005
    #n_iter = 10000
    gc.enable()
    N, n = X.shape
    # Initialize the centers
    c_random = np.random.randint(0,n,k)
    clusters = X[c_random]
    cluster_index = c_random
    # Ciclo principal
    Jdiff = np.inf
    Jprev = np.inf
    i = 0
    while (Jdiff > epsilon): #& (i < n_iter):
        # computar la matrix U
        U = compute_U(clusters, X, metric)
        # computar el costo
        J = cost_function(clusters, X, metric)
        # actualizar los clusters
        clusters = update_cluster(U, X, 3)
        Jdiff = np.abs(J-Jprev)
        Jprev = J
        i += 1
    return clusters

def update_cluster(U, data, k):
    c = np.arange(k).reshape(k,1)
    mask = U == c
    mask = np.expand_dims(mask, axis=-1)
    A = np.expand_dims(data, axis=0)
    mask = np.where(mask, A, np.nan)
    new_cluster = np.nanmean(mask, axis=1)
    return new_cluster

# Compute the matrix U
def compute_U(clusters, data, metric):
    U = cdist(clusters, data, metric=metric).T
    U = np.argmin(U, axis=-1)
    return U

def cost_function(clusters, data, metric):
    A = data.reshape((*data.shape,1))
    B = clusters.reshape((1,*clusters.shape))
    J = cdist(clusters, data, metric=metric)
    J = np.sum(np.min(J, axis=-1))
    return J

def cmeans(X, k, metric):
    n,N = X.shape
    m = 2
    previous_J, diff = np.inf, np.inf
    n_iter = 1000
    epsilon = 0.0005
    # random initialization
    U = rng.rand(k, n)
    U = U / U.sum(axis=0)
    # Ciclo principal
    i = 0
    def update_U(U, data, clusters):
        dist = cdist(clusters, data, metric = metric)
        n_clusters = len(clusters)
        d_ij = dist.reshape(n_clusters, 1, n)
        d_kj = dist.reshape(1, n_clusters, n)

        U = d_ij / d_kj
        U = np.power(U, 2 / (m - 1))
        U = U.sum(axis=1)
        U = 1 / U
        return U

    def cost_function_f(U, data, clusters):
        Um = np.power(U, m)
        dist = cdist(clusters, data, metric = metric)
        J = np.sum(Um * dist)
        return J

    def calculate_cluster(U, data):
        Um = np.power(U, m)
        C = (Um @ data) / Um.sum(axis=1).reshape(k, 1)
        return C
    while (diff > 0) & (previous_J > epsilon) & (i < n_iter):
        # Calculamos los clusters
        clusters = calculate_cluster(U,X)
        # Calculamos el costo
        J = cost_function_f(U, X, clusters)
        # Calculamos la matriz de pertenencia
        U = update_U(U, X, clusters)
        # Criterio de parada
        diff = previous_J - J
        previous_J = J
        i += 1
    return clusters


# Toy dataset
ras = [0.4,0.5,0.7]
data = [i1,i2,ie]
metrics = ['euclidean', 'cosine', 'cityblock']
results = {}

for i in range(len(data)):
    X = data[i]
    for metric in metrics:
        for r_a in ras:
            results[str(i)+str(r_a)+metric] = subtractive(X, r_a, metric)

with open('subtractive_iris.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = [i1,i2,ie]
sigmas = [0.4,0.5,0.7]
metrics = ['euclidean', 'cosine', 'cityblock']
results_m = {}

for i in range(len(data)):
    X = data[i]
    for metric in metrics:
        for sigma in sigmas:
            results_m[str(i)+str(sigma)+metric] = mountain(X, sigma, metric)

with open('mountain_iris.pickle', 'wb') as handle:
    pickle.dump(results_m, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = [i1,i2,ie]
metrics = ['euclidean', 'cosine', 'cityblock']
ks = [[6,5,3,1],[3,2,1],[12,7,4,3]]
results_k = {}
for i in range(len(data)):
    X = data[i]
    for j in range(len(metrics)):
        metric = metrics[j]
        for k in ks[j]:
            a = kmeans(X,k,metric)
            results_k[str(i)+str(metric)+str(k)] = a

with open('kmeans_iris.pickle', 'wb') as handle:
    pickle.dump(results_k, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = [i1,i2,ie]
metrics = ['euclidean', 'cosine', 'cityblock']
ks = [[6,5,3,1],[3,2,1],[12,7,4,3]]
results_c = {}
for i in range(len(data)):
    X = data[i]
    for j in range(len(metrics)):
        metric = metrics[j]
        for k in ks[j]:
            a = cmeans(X,k,metric)
            results_c[str(i)+str(metric)+str(k)] = a

with open('cmeans_iris.pickle', 'wb') as handle:
    pickle.dump(results_c, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Credit card dataset
ras = [0.4,0.5,0.7]
data = [c1,c2,ce]
metrics = ['euclidean', 'cosine', 'cityblock']
results = {}

for i in range(len(data)):
    X = data[i]
    for metric in metrics:
        for r_a in ras:
            results[str(i)+str(r_a)+metric] = subtractive(X, r_a, metric)

with open('subtractive_credit.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = [c1,c2,ce]
sigmas = [0.4,0.5,0.7]
metrics = ['euclidean', 'cosine', 'cityblock']
results_m = {}

for i in range(len(data)):
    X = data[i]
    for metric in metrics:
        for sigma in sigmas:
            results_m[str(i)+str(sigma)+metric] = mountain(X, sigma, metric)

with open('mountain_credit.pickle', 'wb') as handle:
    pickle.dump(results_m, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = [c1,c2,ce]
metrics = ['euclidean', 'cosine', 'cityblock']
ks = [[6,5,3,1],[3,2,1],[12,7,4,3]]
results_k = {}
for i in range(len(data)):
    X = data[i]
    for j in range(len(metrics)):
        metric = metrics[j]
        for k in ks[j]:
            a = kmeans(X,k,metric)
            results_k[str(i)+str(metric)+str(k)] = a

with open('kmeans_credit.pickle', 'wb') as handle:
    pickle.dump(results_k, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = [c1,c2,ce]
metrics = ['euclidean', 'cosine', 'cityblock']
ks = [[6,5,3,1],[3,2,1],[12,7,4,3]]
results_c = {}
for i in range(len(data)):
    X = data[i]
    for j in range(len(metrics)):
        metric = metrics[j]
        for k in ks[j]:
            a = cmeans(X,k,metric)
            results_c[str(i)+str(metric)+str(k)] = a

with open('cmeans_credit.pickle', 'wb') as handle:
    pickle.dump(results_c, handle, protocol=pickle.HIGHEST_PROTOCOL)
