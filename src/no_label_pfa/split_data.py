import numpy as np
import pandas as pd

# Function to calculate the Shannon mutal information of features with the components of the output function
# path_original_data: string path to the original input file
# n_splits: number of splits/output files to generate
# path_labels: string path to the file containing lables for the dataset (e.g. the dbscan output file)
# clusters: list of clusters to be considered in the calculation. If empty, all clusters are considered
# cluster_ratio: ratio between clusters for the splits. Overwrites n_splits
# number_output_functions: Number of output features that are to be modeled, i.e. the number of components of the vector-valued output-function. The values are stored in the first number_output_functions rows of the csv-file.
def split_data(path_original_data, n_splits, path_labels="dbscan_labels.csv",clusters=[], cluster_ratio=[]):

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
        clusters = np.unique(data.T[0])
    
    _, counts = np.unique(data.T[0], return_counts=True)

    split_size = np.ceil((len(data.T)/n_splits)/len(clusters))
    if len(cluster_ratio):
        n_splits = int(np.ceil(len(data.T)/np.sum(cluster_ratio)))
    else:
        cluster_ratio = [int(split_size) for i in range(len(clusters))]

    for i in range(n_splits):
        print("Split "+str(i))
        split = []
        for c in clusters:
            for n in range(cluster_ratio[int(c)]):

                if (i*cluster_ratio[int(c)]) + n >= counts[int(c)]:
                    print("No more datapoints for cluster "+str(c))
                else:
                    split.append(data.values.T[(i*cluster_ratio[int(c)]) + n + np.sum(counts[:int(c)])]) 

        split = np.array(split)        
        np.savetxt("split_"+str(i)+"_y.csv", split.T[0], delimiter=",")
        np.savetxt("split_"+str(i)+".csv", split.T[1:], delimiter=",")

        


            

