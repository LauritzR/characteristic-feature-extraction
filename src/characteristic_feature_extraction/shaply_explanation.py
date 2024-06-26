import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
import shap

# Function to identify the relations between the label and the identified features via a shap explainer
# data: Dataframe containing the unlabeled data
# path_labels: string path to the file containing lables for the dataset (e.g. the dbscan output file)
# path_mutual_information: string path to the file containing the labels and their mutual information
# n_highest_mutual_information:  number of features with the highest mutual information to select. Default value -1 selects all principal features.
# number_sweeps: number of sweeps for training
# feature_selection: PFA = 0, random features = 1 or all features = 2
# clusters: list of clusters to be considered in the calculation. If empty, all clusters are considered
def shaply_explanation(data, path_labels="dbscan_labels.csv", path_mutual_information="mutual_information0.csv", n_highest_mutual_information=-1, number_sweeps=20, feature_selection=0, clusters=[]):
   

    clustering = pd.read_csv(path_labels, sep=',', header=None).to_numpy()

    data_total = pd.DataFrame(np.c_[clustering, data.T])

    if len(clusters) > 0:
        clusters = sorted(clusters)
        drop = [i for i in range(len(data_total)) if data_total.loc[i][0] not in clusters]
        data_total = data_total.drop(drop)
    else:
        clusters = np.unique(clustering)

    data = data_total.sample(frac=0.8)
    data_test = data_total.drop(data.index)
    data = data.transpose().to_numpy()
    data_test = data_test.transpose().to_numpy()

    # Features recommendation from PFA and mutual information
    features_mutual_information_label = pd.read_csv(
        path_mutual_information).sort_values(by=['mutual information'], ascending=False)
    print("Mutual information of system state with itself: ", features_mutual_information_label['mutual information'].iloc[0])

    features_mutual_information_label = features_mutual_information_label.iloc[1:, :]

    print("Mutual information of the first selected feature: ", features_mutual_information_label['mutual information'].iloc[0])
    print("Mutual information of the last selected feature: ", features_mutual_information_label['mutual information'].iloc[n_highest_mutual_information-1])

    if n_highest_mutual_information > 0:
        # Take the featuress with more mutual information than the threshold
        selected_features = features_mutual_information_label[:n_highest_mutual_information]
    else:
        selected_features = features_mutual_information_label

    # List of indices of the rows that are to be taken from the data file and correspond to the selected features
    list_variables = sorted(list(selected_features["index feature"].apply(lambda x: x+1)))

    non_constant_metrics = []
    constant_metrics = []
    for i in range(1, data.shape[0]):
        if max(data[i, :]) > min(data[i, :]):
            non_constant_metrics.append(i)
        else:
            constant_metrics.append(i)
    if feature_selection == 1:
        list_variables = sorted(random.sample(non_constant_metrics,len(list_variables)))
    if feature_selection == 2:
        # Train on the total number of non-constant metrics
        list_variables = sorted(non_constant_metrics)
    len_list_variables = len(list_variables)

    print("Number selected features:")
    print(len_list_variables)

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
    print('balanced accuracy on test set:')
    print(balanced_accuracy_score(y_test, y_pred))
    print('balanced accuracy on training set:')
    print(balanced_accuracy_score(y_train, mlp.predict(X_train_scaled)))

    cm_mlp = confusion_matrix(y_test, y_pred)
    print('confusion matrix on test set:')
    print(cm_mlp)

    explainer = shap.KernelExplainer(mlp.predict_proba, X_train_scaled)
    shap_values = explainer.shap_values(X_train_scaled)

    for idx, s in enumerate(shap_values):
        shap.summary_plot(s, X_train_scaled, feature_names=selected_features['feature name'].to_numpy(), show=False)
        plt.title(int(clusters[idx]))
        plt.show()

