from sklearn.cluster import KMeans


class KMeansCluster:
    def __init__(self, n_clusters=500):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    def generate_pseudo_labels(self, x):
        print("Generating pseudo labels using K-menas clustering.")
        cluster = self.kmeans.fit(x)
        return cluster.labels_
