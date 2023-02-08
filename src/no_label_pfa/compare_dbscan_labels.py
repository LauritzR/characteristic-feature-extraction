import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# compare_dbscan_labels paramters
# path_comparison_labels: string path to the file containing labels to compare
# path_dbscan_labels: string path to the file containing the labels found by dbscan, optional
def compare_dbscan_labels(path_comparison_labels, path_dbscan_labels="dbscan_labels.csv"):
    comparison_labels = pd.read_csv(
        path_comparison_labels, sep=',', header=None).to_numpy().flatten()
    clustering = pd.read_csv(path_dbscan_labels, sep=',',
                             header=None).to_numpy().flatten()

    cluster_bins = []
    for l in np.unique(clustering):
        tmp = [comparison_labels[i]
               for i in range(len(clustering)) if clustering[i] == l]
        cluster_bins.append(tmp)

    plt.ylabel("Count", fontsize=12)
    plt.xlabel("DBSCAN Label", fontsize=12)

    plt.xticks(range(len(np.unique(clustering))))

    n, bins, patches = plt.hist(
        cluster_bins, histtype='barstacked', label=np.unique(comparison_labels))

    plt.legend(title="Comparison labels", fancybox=True)
    plt.show()
