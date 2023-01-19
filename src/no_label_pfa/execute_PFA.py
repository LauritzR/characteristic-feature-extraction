# Copyright with the authors of the publication "A principal feature analysis"

import numpy as np
from principal_feature_analysis import principal_feature_analysis
from parallel_principal_feature_analysis import principal_feature_analysis as par_principal_feature_analysis

def pfa(path, number_sweeps=1, cluster_size=50, alpha=0.01, min_n_datapoints_a_bin=500, shuffle_feature_numbers=0, frac=1, calculate_mutual_information=0, basis_log_mutual_information=2, parallel=False):
    # pf_ds = principal features related to output functions, pf = all principal features
    start_time=time.time()
    number_output_functions=1
    list_pf = []

    # The csv file's content is an m x n Matrix with m - number components of output-function = number features and n = number of data points
    # where the first number components of output-function rows contain the value of the vector-valued output function for each of the n data points
    # e.g. in case of a one-dimensional output function, the first row can be the label for each data point
    data = pd.read_csv(path, sep=',', header=None)
    dummy = np.zeros(data.shape[1])
    data = pd.concat((pd.DataFrame(dummy).T, data), axis=0, ignore_index=True)


    for sweep in range(0,number_sweeps):
        print("Sweep number: " + str(sweep+1))
        pf, pf_s=find_relevant_principal_features(data,number_output_functions,cluster_size,alpha,min_n_datapoints_a_bin,shuffle_feature_numbers,frac, parallel)
        list_pf.append(pf_s)
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


    pf_from_intersection=list_pf[0]
    if number_sweeps > 1:
        for i in range(1, len(list_pf)):
            pf_from_intersection=list(set(pf_from_intersection).intersection(set(list_pf[i])))
        f = open("principal_features_intersection.txt", "w")
        for i in pf_from_intersection:
            f.write(str(i)+str(","))
        f.close()
 
    return pf_from_intersection
