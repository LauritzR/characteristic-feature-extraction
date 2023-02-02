from umap import UMAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def umap(data_path, pf_path, n_components=2, init='random', random_state=0):

    with open(pf_path) as f:
        pfs = f.readlines() 
    pfs = [int(x[:len(x)-2]) for x in pfs]

    X = pd.read_csv(data_path, sep=',', header=None)

    X_pf = X.T[pfs]

    X_embedded = UMAP(n_components=2, init='random', random_state=0).fit_transform(X_pf).T

    np.savetxt("umap_output.csv", X_embedded, delimiter=",")

    plt.scatter(X_embedded[0], X_embedded[1])
    plt.show()
