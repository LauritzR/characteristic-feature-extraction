# Copyright with the authors of the publication "A principal feature analysis"


from .find_relevant_principal_features import find_relevant_principal_features
import time
import pandas as pd
import numpy as np
from configparser import ConfigParser


# paramters for the PFA
# path: string path to the input file
# number_output_functions: Number of output features that are to be modeled, i.e. the number of components of the vector-valued output-function. The values are stored in the first number_output_functions rows of the csv-file.
# number_sweeps: Number of sweeps of the PFA. The result of the last sweep is returned.
# In addition, the return of each sweep are interesected and returned as well.
# cluster_size: number of nodes of a subgraph in the principal_feature_analysis
# alpha=0.01: Level of significance
# min_n_datapoints_a_bin: minimum number of data points for each bin in the chi-square test
# shuffle_feature_numbers: if True the number of the features is randomly shuffled
# frac: the fraction of the dataset that is used for the analysis. The set is randomly sampled from the input csv
# parallel: if True the parallelized version of the PFA is used


def pfa(path, number_sweeps=1, cluster_size=50, alpha=0.01, min_n_datapoints_a_bin=500, shuffle_feature_numbers=0, frac=1, parallel=False):

    config = ConfigParser()

    config["PFA PARAMETERS"] = {
        "min_n_datapoints_a_bin": str(min_n_datapoints_a_bin),
        "alpha": str(alpha)
    }
    with open('config.ini', 'w') as conf:
        config.write(conf)

    # pf_ds = principal features related to output functions, pf = all principal features
    start_time = time.time()
    number_output_functions = 1
    list_pf = []

    # The csv file's content is an m x n Matrix with m - number components of output-function = number features and n = number of data points
    # where the first number components of output-function rows contain the value of the vector-valued output function for each of the n data points
    # e.g. in case of a one-dimensional output function, the first row can be the label for each data point
    data = pd.read_csv(path, sep=',', header=None)
    dummy = np.zeros(data.shape[1])
    data = pd.concat((pd.DataFrame(dummy).T, data), axis=0, ignore_index=True)

    for sweep in range(0, number_sweeps):
        print("Sweep number: " + str(sweep+1))
        pf, pf_s = find_relevant_principal_features(
            data, number_output_functions, cluster_size, alpha, min_n_datapoints_a_bin, shuffle_feature_numbers, frac, parallel)
        list_pf.append(pf)
        f = open("principal_features_structured"+str(sweep)+".txt", "w")
        for i in pf_s:
            for j in i:
                f.write(str(j) + str(","))
            f.write("\n")
        f.close()
        # Output the principal features in a list where the numbers correspond to the rows of the input csv-file
        f = open("principal_features"+str(sweep)+".txt", "w")
        for i in pf:
            f.write(str(i) + str(","))
            f.write("\n")
        f.close()

    print("Time needed for the PFA in seconds: " + str(time.time()-start_time))

    pf_from_intersection = list_pf[0]
    if number_sweeps > 1:
        for i in range(1, len(list_pf)):
            pf_from_intersection = list(
                set(pf_from_intersection).intersection(set(list_pf[i])))
        f = open("principal_features_intersection.txt", "w")
        for i in pf_from_intersection:
            f.write(str(i)+str(","))
        f.close()

    return pf_from_intersection
