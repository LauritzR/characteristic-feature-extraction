import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random


# Function to identify the relations between the label and the identified features via a tree explainer
# path_original_data: string path to the original input file
# path_labels: string path to the file containing lables for the dataset (e.g. the dbscan output file)
# path_mutual_information: string path to the file containing the labels and their mutual information
# n_highest_mutual_information:  number of features with the highest mutual information to select. Default value -1 selects all principal features.
# number_sweeps: number of sweeps for training
# feature_selection: PFA = 0, random features = 1 or all features = 2
# clusters: list of clusters to be considered in the calculation. If empty, all clusters are considered
# max_depth: maximum depth of the tree
# min_samples_leaf: only split if at least min_samples_leaf are left in the resulting left and right node
def tree_explanation(path_original_data, path_labels="dbscan_labels.csv", path_mutual_information="mutual_information0.csv", n_highest_mutual_information=-1, number_sweeps=20, feature_selection=0, clusters=[], max_depth=None, min_samples_leaf=1):
    data = pd.read_csv(path_original_data, sep=",", header=None).transpose()

    clustering = pd.read_csv(path_labels, sep=',', header=None).to_numpy()

    data_total = pd.DataFrame(np.c_[clustering, data])

    if len(clusters) > 0:
        drop = [i for i in range(len(data_total)) if data_total.loc[i][0] not in clusters]
        data_total = data_total.drop(drop)

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

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print('Score on test and train data with the PFA metrics')
    print('r2-accuracy on test set:')
    print(clf.score(X_test_scaled, y_test))
    print('r2-accuracy on training set:')
    print(clf.score(X_train_scaled, y_train))
    print('balanced accuracy on test set:')
    print(balanced_accuracy_score(y_test, y_pred))
    print('balanced accuracy on training set:')
    print(balanced_accuracy_score(y_train, clf.predict(X_train_scaled)))

    cm_clf = confusion_matrix(y_test, y_pred)
    print('confusion matrix on test set:')
    print(cm_clf)

    plot_tree(clf, feature_names=selected_features['feature name'].values, proportion=True, class_names=clf.classes_.astype(str))
    plt.show()
