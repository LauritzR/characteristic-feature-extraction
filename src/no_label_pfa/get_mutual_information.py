# Copyright with the authors of the publication "A principal feature analysis"

import numpy as np
import pandas as pd
import math
from configparser import ConfigParser

# Function to calculate the Shannon mutal information of features with the components of the output function
# path_original_data: string path to the original input file
# path_principal_features_cluster_differences: string path to the txt file containing the principal features detected by find_cluster_differences
# path_labels: string path to the file containing lables for the dataset (e.g. the dbscan output file)
# path_feature_names: path to csv containing the names of the features. The column containing the names must be named 'feature name'
# clusters: list of clusters to be considered in the calculation. If empty, all clusters are considered
# number_output_functions: Number of output features that are to be modeled, i.e. the number of components of the vector-valued output-function. The values are stored in the first number_output_functions rows of the csv-file.
# basis_log_mutual_information:  the basis for the logarithm used to calculate the mutual information.

def get_mutual_information(path_original_data, path_principal_features_cluster_differences="principal_features_cluster_differences.txt", path_labels="dbscan_labels.csv", path_feature_names="",clusters=[], number_output_functions=1, basis_log_mutual_information=2):

    with open(path_principal_features_cluster_differences) as f:
        pfs = f.readlines()
    list_variables = [int(x[:len(x)-2])+number_output_functions for x in pfs]

    try:
        feature_names = pd.read_csv(path_feature_names, sep=',', header=0)['feature name'].to_numpy()[list_variables]
    except:
        feature_names = [str(e) for e in list_variables]

    config = ConfigParser()
    config.read("config.ini")
    min_n_datapoints_a_bin = config.getint(
        "PFA PARAMETERS", "min_n_datapoints_a_bin")

    data = pd.read_csv(path_original_data, sep=',', header=None)
    clustering = pd.read_csv(path_labels, sep=',', header=None).to_numpy()
    data = pd.DataFrame(np.c_[clustering, data.T].T)


    if len(clusters) > 0:
        drop = [i for i in range(len(data.iloc[0]))
                if data.iloc[0][i] not in clusters]
        data.drop(columns=drop, inplace=True)

    # Calulate the Shannon mutual information
    def make_summand_from_frequencies(x, y):
        if x == 0:
            return 0
        else:
            return x * math.log2(x / y) / math.log2(basis_log_mutual_information)

    # Insert the the indices of the rows where the components of the output functions are stored
    for i in range(0, number_output_functions):
        list_variables.insert(i, i)

        print(feature_names)
        feature_names = np.insert(feature_names, i, "Label {}".format(i))
        print(feature_names)

    data_init = data.to_numpy()
    data = data_init[list_variables, :]


    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_features = [i for i in range(0, m)]
    constant_features = []

    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print(
                "Feature #"f"{list_variables[i]}" " has only constant values")
            left_features.remove(i)
            constant_features.append(list_variables[i])
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                # if end of the data points leave the for-loop
                if point >= (datapoints.size - 1):
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
            if len(list_points_of_support) > 0:
                # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                if list_points_of_support[0] > datapoints[0]:
                    list_points_of_support.insert(0, datapoints[0])
            else:
                list_points_of_support.append(datapoints[0])
            # Add last point of support such that last data point is included (half open interals in Python!)
            list_points_of_support.append(datapoints[-1] + 0.1)
            # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
            if datapoints[datapoints >= list_points_of_support[-2]].size < min_n_datapoints_a_bin:
                # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                if len(list_points_of_support) > 2:
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]

    # Check for constant features
    if constant_features != []:
        print("List of features with constant values:")
        print(constant_features)
    for id_output in range(0, number_output_functions):
        # Warn if the output function is constant e.g. due to an unsuitable binning
        if id_output in constant_features or len(freq_data[id_output]) < 2:
            print("Warning: Output function " +
                  str(id_output) + " is constant!")

    # Calculate the mutual information for each feature with the corresponding component of the output function
    list_of_data_frames = []
    # number of featuers plus one component of the output-function
    mutual_info = np.ones(
        (1, len(left_features) - number_output_functions + 1))
    for i in range(0, number_output_functions):
        list_of_features = list(
            range(number_output_functions, len(left_features)))
        list_of_features.insert(0, i)
        id_features = np.array(list_variables)[left_features]
        id_features = [x-number_output_functions for x in id_features]

        for j in list_of_features:
            freq_data_product = ((np.histogram2d(
                data[i, :], data[left_features[j], :], bins=(l[i], l[left_features[j]]))[0])) / n
            expfreq = (
                np.outer(freq_data[i], freq_data[left_features[j]])) / (n * n)
            if j < number_output_functions:
                mutual_info[0, 0] = np.sum(np.array(list(map(
                    make_summand_from_frequencies, freq_data_product.flatten().tolist(), expfreq.flatten().tolist()))))
            else:
                mutual_info[0, j-number_output_functions+1] = np.sum(np.array(list(map(
                    make_summand_from_frequencies, freq_data_product.flatten().tolist(), expfreq.flatten().tolist()))))
        pd_mutual_information = pd.DataFrame(
            {"index feature": id_features, "mutual information": mutual_info.tolist()[0], "feature name": feature_names})
        pd_mutual_information['index feature'] = pd_mutual_information['index feature'].astype(
            int)
        list_of_data_frames.append(pd_mutual_information)

    for i in range(len(list_of_data_frames)):
        list_of_data_frames[i].to_csv("mutual_information"+str(i)+".csv")

    return list_of_data_frames
