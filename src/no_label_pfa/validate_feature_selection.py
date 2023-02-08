import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import random

# Function to calculate the Shannon mutal information of features with the components of the output function
# path_original_data: string path to the original input file
# path_labels: string path to the file containing lables for the dataset (e.g. the dbscan output file)
# path_mutual_information: string path to the file containing the labels and their mutual information
# n_highest_mutual_information:  number of features with the highest mutual information to select. Default value -1 selects all principal features.
# number_sweeps: number of sweeps for training
def validate_feature_selection(path_original_data, path_labels="dbscan_labels.csv", path_mutual_information="mutual_information0.csv", n_highest_mutual_information=-1, number_sweeps=20):
    gene_selection = 0  # validate on PFA = 0, random genes = 1 or all genes = 2

    data = pd.read_csv(path_original_data, sep=",", header=None).transpose()

    clustering = pd.read_csv(path_labels, sep=',', header=None).to_numpy()

    data_total = pd.DataFrame(np.c_[clustering, data])

    data = data_total.sample(frac=0.8)
    data_test = data_total.drop(data.index)
    data = data.transpose().to_numpy()
    data_test = data_test.transpose().to_numpy()

    # Gene recommendation from PFA and mutual information
    genes_mutual_information_label = pd.read_csv(
        path_mutual_information).sort_values(by=['mutual information'], ascending=False)
    print("Mutual information of system state with itself: ",
          genes_mutual_information_label['mutual information'].iloc[0])

    genes_mutual_information_label = genes_mutual_information_label.iloc[1:, :]

    print("Mutual information of the first selected feature: ",
          genes_mutual_information_label['mutual information'].iloc[0])
    print("Mutual information of the last selected feature: ",
          genes_mutual_information_label['mutual information'].iloc[n_highest_mutual_information-1])

    if n_highest_mutual_information > 0:
        # Take the genes with more mutual information than the threshold
        selected_genes = genes_mutual_information_label[:n_highest_mutual_information]
    else:
        selected_genes = genes_mutual_information_label

    print()
    # List of indices of the rows that are to be taken from the data file and correspond to the selected features
    list_variables = sorted(
        list(selected_genes["index feature"].apply(lambda x: x+1)))

    r2_train = np.zeros((1, number_sweeps))
    r2_test = np.zeros((1, number_sweeps))
    number_wrongly_classified = np.zeros((1, number_sweeps))

    for sweep in range(0, number_sweeps):
        if sweep <= 0:
            non_constant_metrics = []
            constant_metrics = []
            for i in range(1, data.shape[0]):
                if max(data[i, :]) > min(data[i, :]):
                    non_constant_metrics.append(i)
                else:
                    constant_metrics.append(i)
        if gene_selection == 1:
            list_variables = sorted(random.sample(non_constant_metrics,
                                                  len(list_variables)))
        if gene_selection == 2:
            # Train on the total number of non-constant metrics
            list_variables = sorted(non_constant_metrics)
        len_list_variables = len(list_variables)
        if sweep <= 0:
            print("Number selected genes:")
            print(len_list_variables)

        print("Sweep #" + str(sweep))
        X_train = data[list_variables, :].transpose()
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        y_train = data[0, :]
        X_test = data_test[list_variables, :].transpose()
        X_test_scaled = scaler.transform(X_test)
        y_test = data_test[0, :]

        mlp = MLPClassifier(max_iter=8000000)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
        print('Score on test and train data with the PFA metrics')
        print('r2-accuracy on test set:')
        print(mlp.score(X_test_scaled, y_test))
        print('r2-accuracy on training set:')
        print(mlp.score(X_train_scaled, y_train))
        cm_mlp = confusion_matrix(y_test, y_pred)

        r2_test[0, sweep] = accuracy_score(y_test, y_pred)
        number_wrongly_classified[0, sweep] = cm_mlp[1, 0]+cm_mlp[0, 1]
        r2_train[0, sweep] = mlp.score(X_train_scaled, y_train)

    print('\n')
    if gene_selection == 0:
        print("Results on PFA gene selection:")
    if gene_selection == 1:
        print("Results on randomly selected genes:")
    if gene_selection == 2:
        print("Results on all non-constant genes:")

    print("r2_test mean: " + str(r2_test.mean()))
    print("r2_train mean: " + str(r2_train.mean()))
    print("r2_test std: " + str(r2_test.std()))
    print("max r2_test: " + str(max(r2_test[0, :])))
    print("Wrongly classified mean: " + str(number_wrongly_classified.mean()))
    print("Wrongly classified std: " + str(number_wrongly_classified.std()))
