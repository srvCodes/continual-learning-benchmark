from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def kmeans_sample(X, k):
    kmeans = KMeans(n_clusters=k, random_state=10).fit(X)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    return closest.tolist()
