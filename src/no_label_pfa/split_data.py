import numpy as np
import pandas as pd

# Function to split a data set into several sub data sets to check results for senesitivity of the data set
# path_original_data: string path to the original input file
# n_splits: number of splits/output files to generate
# path_labels: string path to the file containing lables for the dataset (e.g. the dbscan output file)
# clusters: list of clusters to be considered in the calculation. If empty, all clusters are considered
# number_per_cluster: number of samples per cluster in each splitted file
def split_data(path_original_data, n_splits, path_labels="dbscan_labels.csv",clusters=[], number_per_cluster=[]):

    data = pd.read_csv(path_original_data, sep=',', header=None).to_numpy().T
    clustering = pd.read_csv(path_labels, sep=',', header=None).to_numpy()
    data = pd.DataFrame(np.c_[clustering, data].T)
    data = data.sort_values(data.columns[0],ascending=True)

    print(data.shape)
    if len(clusters) > 0:
        drop = [i for i in range(len(data.iloc[0]))
                if data.iloc[0][i] not in clusters]
        data.drop(columns=drop, inplace=True)
    else:
        clusters = np.unique(clustering)
    
    _, counts = np.unique(data.T[0], return_counts=True)

    split_size = np.ceil((len(data.T)/n_splits)/len(clusters))

    if not len(number_per_cluster):
        number_per_cluster = [int(split_size) for i in range(len(clusters))]

    for i in range(n_splits):
        print("Split "+str(i))
        split = []
        for c in clusters:
            for n in range(number_per_cluster[int(c)]):

                if (i*number_per_cluster[int(c)]) + n >= counts[int(c)]:
                    print("No more datapoints for cluster "+str(c))
                else:
                    split.append(data.values.T[(i*number_per_cluster[int(c)]) + n + np.sum(counts[:int(c)])]) 

        split = np.array(split)        
        np.savetxt("split_"+str(i)+"_y.csv", split.T[0], delimiter=",")
        np.savetxt("split_"+str(i)+".csv", split.T[1:], delimiter=",")
