from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dbscan paramters
# path_embedding: string path to the embedding output file (e.g. tsne or umap)
# eps: maximum distance between two samples for them to be considered neighbours
# min_samples: number of samples in a neighborhood for a point to be considered as a core point
# plot: flag for visual output
# cmap: colormap used for plotting (default is 'viridis', other examples are 'PiYG', 'twilight' or 'tab20'. For more details check the matplotlib colomap documentation)

def dbscan(path_embedding, eps=2, min_samples=15, plot=True, cmap=None):

    X = pd.read_csv(path_embedding, sep=',', header=None).to_numpy().T

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    np.savetxt("dbscan_labels.csv", clustering, delimiter=",")

    values, counts = np.unique(clustering, return_counts=True)

    for v, c in zip(values, counts):
        print("Label {}: {}".format(v, c))

    if plot:
        plt.figure(figsize=(10,6))
        scatter = plt.scatter(X[0], X[1], cmap=cmap, c=clustering)

        plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()
