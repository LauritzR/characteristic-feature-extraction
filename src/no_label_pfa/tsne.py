from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# tsne paramters
# path_original_data: string path to the original input file
# path_principal_features: string path to the txt file containing the principal features
# n_components: number of components in the embedding output
# learning_rate: learning rate of the tsne. Can be a float between 10.0 and 1000.0. Default 'auto'
# init: initialization of embedding. Default 'random'
# perplexity: related to the number of nearest neighbors that is used in other manifold learning algorithms. Consider a value between 5 and 50.
# plot: flag for visual output

def tsne(path_original_data, path_principal_features="principal_features0.txt", n_components=2, learning_rate='auto', init='random', perplexity=30, plot=True):

    with open(path_principal_features) as f:
        pfs = f.readlines()
    pfs = [int(x[:len(x)-2]) for x in pfs]

    X = pd.read_csv(path_original_data, sep=',', header=None)

    X_pf = X.T[pfs]

    X_embedded = TSNE(n_components=n_components, learning_rate=learning_rate,
                      init=init, perplexity=perplexity).fit_transform(X_pf).T

    np.savetxt("tsne_output.csv", X_embedded, delimiter=",")

    if plot:
        plt.scatter(X_embedded[0], X_embedded[1])
        plt.show()
