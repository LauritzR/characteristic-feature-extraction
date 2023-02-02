from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dbscan(path_embedded, path_original, path_pfs="principal_features0.txt", eps=2, min_samples=15):
    X = pd.read_csv(path_embedded, sep=',', header=None).to_numpy().T

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    np.savetxt("dbscan_labels.csv", clustering, delimiter=",")

    with open(path_pfs) as f:
        pfs = f.readlines() 
    pfs = [int(x[:len(x)-2]) for x in pfs]

    pf_data = pd.read_csv(path_original, sep=',', header=None).to_numpy()[pfs].T

    data = np.c_[clustering, pf_data]

    np.savetxt("dbscan_output.csv", data.T, delimiter=",")

    for i in np.unique(clustering):
        mask = clustering == i
        plt.scatter(X[mask, 0], X[mask, 1],  label=i)

    values, counts = np.unique(clustering, return_counts=True)

    for v,c in zip(values, counts):
        print("Label {}: {}".format(v,c))
    plt.legend()
    plt.show()
