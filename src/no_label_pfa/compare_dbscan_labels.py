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
        tmp = [int(l)]
        for cl in np.unique(comparison_labels):
            count = 0
            for i in range(len(comparison_labels)):
                if clustering[i] == l and comparison_labels[i] == cl:
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
