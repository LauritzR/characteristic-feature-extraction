from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def tsne(data_path, pf_path, n_components=2, learning_rate='auto',init='random', perplexity=30):

    with open(pf_path) as f:
        pfs = f.readlines()
    
    pfs = [int(x[:len(x)-2]) for x in pfs]
    X = pd.read_csv(data_path, sep=',', header=None)
    X_pf = X.T[pfs]
    X_embedded = TSNE(n_components=n_components, learning_rate=learning_rate,init=init, perplexity=perplexity).fit_transform(X_pf).T

    np.savetxt("tsne_output.csv", X_embedded, delimiter=",")

    plt.scatter(X_embedded[0], X_embedded[1])
    plt.show()
