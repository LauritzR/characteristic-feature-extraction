# Copyright with the authors of the publication "A principal feature analysis"

import numpy as np
from principal_feature_analysis import principal_feature_analysis
from parallel_principal_feature_analysis import principal_feature_analysis as par_principal_feature_analysis

def find_relevant_principal_features(data,number_output_functions,cluster_size,alpha,min_n_datapoints_a_bin,shuffle_feature_numbers,frac, parallel):
    # In this function the binning is done, the graph is dissected and the relevant features/variables are detected
    if frac<1: # if frac<1 the feature analysis is done only the fraction of the data, randomly sampled
        data=data.sample(frac=frac,axis='columns',replace=False)
    data=data.to_numpy()
    
    m = data.shape[0] #number features
    n = data.shape[1] #number of data points
    l = [0 for i in range(0, m)]  # list of lists with the points of support for the binning
    freq_data = [[0] for i in range(0, m)] # list of histograms
    left_features = [i for i in range(0, m)]  # list of features that is step by step reduced to the relevant ones
    constant_features = []

    # remove constant features and binning (discretizing the continuous values of our features)
    for i in range(0, m): 
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #"f"{i}" " has only constant values")
            left_features.remove(i)
            constant_features.append(i)
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            if len(list_points_of_support) > 0: # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
                if list_points_of_support[0] > datapoints[0]: # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                    list_points_of_support.insert(0, datapoints[0])
            else:
                list_points_of_support.append(datapoints[0])
            list_points_of_support.append(datapoints[-1] + 0.1) # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[-2]].size < min_n_datapoints_a_bin: # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:     # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
        
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]
    print("Binning done!")
    print("List of features with constant values:")
    print(constant_features)
    print("Number of constant features:")
    print(len(constant_features))
    

    print("Starting principal feature analysis!")
    # list_principal_features = list of resulting principal features
    # smaller_than5 = percent of chi-square tests with bins less than 5 data points
    # smaller_than1 = percent of chi-square tests with bins less than 1 data point

    if parallel:
        list_principal_features, smaller_than5, smaller_than1 = par_principal_feature_analysis(cluster_size, data, number_output_functions, freq_data, l, left_features, alpha, shuffle_feature_numbers)
    else:
        list_principal_features, smaller_than5, smaller_than1 = principal_feature_analysis(cluster_size, data, number_output_functions, freq_data, l, left_features, alpha, shuffle_feature_numbers)

    # Assign global index to each principal feature
    list_principal_features_global_indices_structured = []
    for i in list_principal_features:
        intermediate_list = []
        for j in i:
            if len(freq_data[left_features[j+number_output_functions]]) > 1:
                intermediate_list.append(left_features[j+number_output_functions]-number_output_functions)
        if len(intermediate_list) > 0:
            list_principal_features_global_indices_structured.append(intermediate_list)

    print('principal features:')
    print(list_principal_features_global_indices_structured)


    list_principal_features_global_indices = [pf for sublist in list_principal_features_global_indices_structured for pf in sublist]

    return list_principal_features_global_indices, list_principal_features_global_indices_structured
