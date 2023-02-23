import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# compare_dbscan_labels paramters
# path_comparison_labels: string path to the file containing labels to compare
# path_dbscan_labels: string path to the file containing the labels found by dbscan, optional
# clusters: list of dbscan clusters to be considered in the calculation. If empty, all clusters are considered
def compare_dbscan_labels(path_comparison_labels, path_dbscan_labels="dbscan_labels.csv", clusters=[]):
    comparison_labels = pd.read_csv(
        path_comparison_labels, sep=',', header=None).to_numpy().flatten()
    clustering = pd.read_csv(path_dbscan_labels, sep=',',
                             header=None).to_numpy().flatten()

    if len(clusters) == 0:
        clusters = np.unique(clustering)

    cluster_bins = []

    for l in clusters:
        tmp = [int(l)]
        for cl in np.unique(comparison_labels):
            count = 0
            for i in range(len(comparison_labels)):
                if clustering[i] == l and comparison_labels[i] == cl and clustering[i] in clusters:
                    count += 1
            tmp.append(count)

        cluster_bins.append(tmp)

    columns = ['DBSCAN Label']

    for cl in np.unique(comparison_labels):
        columns.append(str(cl))

    df = pd.DataFrame(cluster_bins, columns=columns)

    df.plot(x='DBSCAN Label', kind='bar', stacked=True)
    plt.legend(title="Comparison Label", fancybox=True)
    plt.show()
