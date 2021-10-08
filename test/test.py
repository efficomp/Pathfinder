__author__ = 'Alberto Ortega'
__copyright__ = 'Copyright 2021, ACOFS'
__credits__ = 'Juan José Escobar, Julio Ortega, Jesús González, Miguel Damas'
__license__ = 'GNU GPL v3.0'
__version__ = '2.0'
__maintainer__ = 'Alberto Ortega'
__email__ = 'aoruiz@ugr.es'

import numpy as np
import scipy.io as sp
import sys
import os
sys.path.append(os.path.abspath('.'))
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score
from pathfinder.featureselector import FeatureSelector

if __name__ == "__main__":

    # TIME START
    start_time = time.time()

    # FILE STRINGS
    dataset_X_name           = sys.argv[2]
    dataset_y_name           = sys.argv[3]
    prediction_X_name        = sys.argv[4]
    prediction_y_name        = sys.argv[5]

    # LOADING DATASET FROM MATLAB FILE
    dic_training_data        = sp.loadmat(dataset_X_name)
    dic_training_target_data = sp.loadmat(dataset_y_name)
    dic_testing_data         = sp.loadmat(prediction_X_name)
    dic_testing_target_data  = sp.loadmat(prediction_y_name)

    X_train = np.array(dic_training_data["data"]) 
    y_train = np.reshape(np.array(dic_training_target_data["class"]),len(X_train)) - 1
    X_test  = np.array(dic_testing_data["data"])
    y_test  = np.reshape(np.array(dic_testing_target_data["class"]),len(X_test)) - 1

    # Preprocessing data: Scikit-learn needs Gaussian with zero mean and unit variance.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ACO CALL
    number_ants = int(sys.argv[6])
    number_elitist_ants = int(sys.argv[7])
    iter = int(sys.argv[8])
    version = sys.argv[1]
    final_subset = []
    optimizer = FeatureSelector(version, X_train_scaled, y_train, X_test_scaled, y_test, number_ants, number_elitist_ants, iter, 0.5, 0.5, 0.5, 0.2, 0.2)
    final_subset = optimizer.acoFS()

    # BUILDING SUBSET DATASET
    X_train_subset = X_train[:,final_subset]
    X_test_subset = X_test[:,final_subset]

    # NEURAL NETWORK CLASSIFIER
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    e = 500
    set_sum_classification = 0
    subset_sum_classification = 0
    first_classification = 0
    conf_matrix = 0

    for i in range(10):
        nn_classifier_set = tensorflow.keras.models.Sequential([])
        nn_classifier_set.add(tensorflow.keras.layers.Dense(64, input_dim=len(X_train[0]), activation='sigmoid'))
        nn_classifier_set.add(tensorflow.keras.layers.Dense(3, activation='softmax'))
        nn_classifier_set.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        nn_classifier_set.fit(X_train, y_train, epochs=e, verbose=0)
        train_loss, nn_classifier_set_score = nn_classifier_set.evaluate(X_test, y_test, verbose=0)
        set_sum_classification += nn_classifier_set_score
    set_mean_classification = set_sum_classification/10


    for i in range(10):
        nn_classifier_subset = tensorflow.keras.models.Sequential()
        nn_classifier_subset.add(tensorflow.keras.layers.Dense(64, input_dim=len(X_train_subset[0]), activation='sigmoid'))
        nn_classifier_subset.add(tensorflow.keras.layers.Dense(3, activation='softmax'))
        nn_classifier_subset.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        nn_classifier_subset.fit(X_train_subset,y_train, epochs=e, verbose=0)
        train_loss, nn_classifier_subset_score = nn_classifier_subset.evaluate(X_test_subset, y_test, verbose=0)
        if i==0:
            first_classification = nn_classifier_subset_score
        subset_sum_classification += nn_classifier_subset_score
        predictions = nn_classifier_subset.predict(X_test_subset)
        predicted_labels = np.argmax(predictions, axis=-1)
        conf_matrix += confusion_matrix(y_test, predicted_labels, labels=[0, 1, 2])
    subset_mean_classification = subset_sum_classification/10
    conf_matrix = conf_matrix/10

    clf = svm.SVC()
    cvs_set = np.mean(cross_val_score(clf, X_train, y_train, cv=10))
    cvs_subset = np.mean(cross_val_score(clf, X_train_subset, y_train, cv=10))

    # TIME STOP
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    # OUTPUT
    print("Elapsed time:")
    print("\t", elapsed_time, "seconds")
    print("Number of features:")
    print("\tSet   :", len(X_train[0]))
    print("\tSubset:", len(X_train_subset[0]))
    print("Accuracy of a neural network classifier:")
    print("\tSet    mean accuracy               :", "{:.5f}".format(nn_classifier_set_score*100), "%" )
    print("\tSubset mean classification accuracy:", "{:.5f}".format(subset_mean_classification*100), "%" )
    print("\tSubset 1st classification accuracy :", "{:.5f}".format(first_classification*100), "%" )
    print("\tConfusion matrix:")
    print("\t\t\t", conf_matrix[0])
    print("\t\t\t", conf_matrix[1])
    print("\t\t\t", conf_matrix[2])
    print("\tSet    K-Fold Cross-Validation:", "{:.5f}".format(cvs_set*100), "%")
    print("\tSubset K-Fold Cross-Validation:", "{:.5f}".format(cvs_subset*100), "%")
