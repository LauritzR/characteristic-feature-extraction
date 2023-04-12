import hdbscan as hdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# dbscan paramters
# path_embedding: string path to the embedding output file (e.g. tsne or umap)
# min_cluster_size: smallest size grouping to be considered a cluster
# plot: flag for visual output
# cmap: colormap used for plotting

def hdbscan(path_embedding, min_cluster_size=5, plot=True, cmap=None):

    X = pd.read_csv(path_embedding, sep=',', header=None).to_numpy()

    clusterer = hdb.HDBSCAN(min_cluster_size=min_cluster_size)
    clustering = clusterer.fit_predict(X.T)

    np.savetxt("hdbscan_labels.csv", clustering, delimiter=",")

    values, counts = np.unique(clustering, return_counts=True)

    for v, c in zip(values, counts):
        print("Label {}: {}".format(v, c))

    if plot:
        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(X[0], X[1], cmap=cmap, c=clustering)

        plt.legend(*scatter.legend_elements(), fancybox=True,
                   bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()
