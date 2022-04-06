__author__ = 'Alberto Ortega'
__copyright__ = 'Pathfinder (c) 2021 EFFICOMP'
__credits__ =  'Spanish Ministerio de Ciencia, Innovacion y Universidades under grant number PGC2018-098813-B-C31. European Regional Development Fund (ERDF).'
__license__ = ' GPL-3.0'
__version__ = "2.0"
__maintainer__ = 'Alberto Ortega'
__email__ = 'aoruiz@ugr.es'

import sys
import numpy as np
import pandas as pd
import scipy.io as sp
import time
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from pathfinder.ant import Ant
np.set_printoptions(threshold=sys.maxsize)

 
class FeatureSelector:
    """Class for Ant System Optimization algorithm designed for Feature Selection.

    :param dtype: Format of the dataset.
    :param data_training_name: Path to the training data file (mat) or path to the dataset file (csv).
    :param class_training: Path to the training classes file (mat).
    :param data_testing: Path to the testing data file (mat).
    :param class_testing: Path to the testing classes file (mat).
    :param numberAnts: Number of ants of the colonies.
    :param iterations: Number of colonies of the algorithm.
    :param n_features: Number of features to be selected.
    :param alpha: Parameter which determines the weight of tau.
    :param beta: Parameter which determines the weight of eta.
    :param Q_constant: Parameter for the pheromones update function.
    :param initialPheromone: Initial value for the pheromones.
    :param evaporationRate: Rate of the pheromones evaporation.
    :type dtype: MAT or CSV
    :type data_training_name: Numpy array
    :type class_training: Numpy array
    :type data_testing: Numpy array
    :type class_testing: Numpy array
    :type numberAnts: Integer
    :type iterations: Integer
    :type n_features: Integer
    :type alpha: Float
    :type beta: Float
    :type Q_constant: Float
    :type initialPheromone: Float
    :type evaporationRate: Float

    """

    def __init__(self, dtype="mat", data_training_name=None, class_training_name=None, numberAnts=1, iterations=1, n_features=1, data_testing_name=None, class_testing_name=None, alpha=1, beta=1, Q_constant=1, initialPheromone=1.0, evaporationRate=0.1):
        """Constructor method.
        """
        time_dataread_start = time.time()
        if dtype=="mat":
            dic_data_training  = sp.loadmat(data_training_name)
            dic_class_training = sp.loadmat(class_training_name)
            self.data_training  = np.array(dic_data_training["data"]) 
            self.class_training = np.reshape(np.array(dic_class_training["class"]),len(self.data_training)) - 1
            
            dic_data_testing  = sp.loadmat(data_testing_name)
            dic_class_testing = sp.loadmat(class_testing_name)
            self.data_testing  = np.array(dic_data_testing["data"]) 
            self.class_testing = np.reshape(np.array(dic_class_testing["class"]),len(self.data_testing)) - 1

            # Free dictionaries memory
            del dic_data_training
            del dic_class_training

        elif dtype=="csv":
            df = pd.read_csv(data_training_name)
            df = df.to_numpy() 
            classes = df[:, -1].astype(int)
            df = np.delete(df, -1, 1)
            self.data_training, self.data_testing, self.class_training, self.class_testing = train_test_split(df, classes, random_state=42)


        #print("Samples x features:", np.shape(self.dataset))

        scaler = StandardScaler().fit(self.data_training)
        self.data_training = scaler.transform(self.data_training)
        self.data_testing = scaler.transform(self.data_testing)

        self.number_ants = numberAnts
        self.ants = [Ant() for _ in range(self.number_ants)]
        self.number_features = len(self.data_training[0])
        self.iterations = iterations
        self.initial_pheromone = initialPheromone
        self.evaporation_rate = evaporationRate
        self.alpha = alpha
        self.beta = beta
        self.Q_constant = Q_constant
        self.feature_pheromone = np.full(self.number_features, self.initial_pheromone)
        self.unvisited_features = np.arange(self.number_features)
        self.ant_accuracy = np.zeros(self.number_ants)
        self.n_features = n_features
        if self.n_features > self.number_features:
            self.n_features = self.number_features
        #random.seed(1) ########################################################################
        
        time_dataread_stop = time.time()
        self.time_dataread = time_dataread_stop - time_dataread_start
        self.time_LUT = 0
        self.time_reset = 0
        self.time_localsearch = 0
        self.time_pheromonesupdate = 0

    def defineLUT(self):
        """Defines the Look-Up Table (LUT) for the algorithm.
        """
        time_LUT_start = time.time()

        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(self.data_training, self.class_training)
        self.LUT = fs.scores_
        sum = np.sum(self.LUT)
        for i in range(len(fs.scores_)):
            self.LUT[i] = self.LUT[i]/sum
        
        time_LUT_stop = time.time()
        self.time_LUT = self.time_LUT + (time_LUT_stop - time_LUT_start)

    def redefineLUT(self, feature): 
        """Re-defines the Look-Up Table (LUT) for the algorithm.
        """
        time_LUT_start = time.time()
        
        weightprob = self.LUT[feature]
        self.LUT[feature] = 0
        mult = 1/(1-weightprob)
        self.LUT = self.LUT * mult
        
        time_LUT_stop = time.time()
        self.time_LUT = self.time_LUT + (time_LUT_stop - time_LUT_start)
        
    def resetInitialValues(self):
        """Initialize the ant array and assign each one a random initial feature.
        """
        time_reset_start = time.time()

        self.ants = [Ant() for _ in range(self.number_ants)]
        initialFeaturesValues = np.arange(self.number_features)
        for i in range(self.number_ants):
            rand = np.random.choice(initialFeaturesValues, 1, p=self.LUT)[0]
            self.ants[i].feature_path.append(rand)
            actual_features_list = self.ants[i].feature_path                     
            actual_subset = np.array(self.data_training[:,actual_features_list])
            actual_classifier = KNeighborsClassifier()
            actual_classifier.fit(actual_subset, self.class_training)
            scores = cross_val_score(actual_classifier, actual_subset, self.class_training, cv=5)
            actual_accuracy = scores.mean()
            np.put(self.ant_accuracy, i, actual_accuracy)

        time_reset_stop = time.time()
        self.time_reset = self.time_reset + (time_reset_stop - time_reset_start)

    def antBuildSubset(self, index_ant):
        """Global and local search for the ACO algorithm. It completes the subset of features of the ant searching.

        :param index_ant: Ant that is going to do the local search.
        :type index_ant: Integer
        """
        time_localsearch_start = time.time()

        # Initialize unvisited features and it removes the first of the ant actual subset
        self.unvisited_features = np.arange(self.number_features)
        indexes = np.where(np.in1d(self.unvisited_features, self.ants[index_ant].feature_path))[0]
        self.unvisited_features = np.delete(self.unvisited_features, indexes)
        self.defineLUT()

        n = 1
        while n < self.n_features:
            # Initialize parameters
            p = np.zeros(np.size(self.unvisited_features))
            p_num = np.zeros(np.size(self.unvisited_features))

            # Compute eta, tau and the numerator for each unvisited feature 
            for index_uf in range(len(self.unvisited_features)):
                eta = self.LUT[self.unvisited_features[index_uf]]
                tau = self.feature_pheromone[index_uf]
                np.put(p_num, index_uf, (tau**self.alpha) * (eta**self.beta))

            den = np.sum(p_num)
            for index_uf in range(len(self.unvisited_features)):
                p[index_uf] = p_num[index_uf] / den
            next_feature = np.random.choice(self.unvisited_features, 1, p=p)[0]
            
            if (n==self.n_features-1):
                new_features_list = np.array(self.ants[index_ant].feature_path)
                new_features_list = np.append(new_features_list,next_feature)
                new_subset = np.array(self.data_training[:,new_features_list])
                new_classifier = KNeighborsClassifier()
                new_classifier.fit(new_subset, self.class_training)
                scores = cross_val_score(new_classifier, new_subset, self.class_training, cv=5)
                new_accuracy = scores.mean()

        
            # Choose the feature with best probability and add to the ant subset
            self.ants[index_ant].feature_path.append(next_feature)
            # Remove the chosen feature of the unvisited features
            self.unvisited_features = np.delete(self.unvisited_features, np.where( self.unvisited_features == next_feature))
            if (n==self.n_features-1):
                np.put(self.ant_accuracy, index_ant, new_accuracy)
            self.redefineLUT(next_feature)
            n=n+1
        
        time_localsearch_stop = time.time()
        self.time_localsearch = self.time_localsearch + (time_localsearch_stop - time_localsearch_start)

        
    def updatePheromones(self):
        """Update the pheromones trail depending on which variant of the algorithm it is selected.
        """
        time_pheromonesupdate_start = time.time()

        for f in self.ants[np.argmax(self.ant_accuracy)].feature_path:
            sum_delta = 0
            sum_delta += self.Q_constant / ((1-self.ant_accuracy[np.argmax(self.ant_accuracy)])*100)

            updated_pheromone = ( 1 - self.evaporation_rate) * self.feature_pheromone[f] + sum_delta
            if(updated_pheromone < 0.4):
                updated_pheromone = 0.4
            #print(updated_pheromone)            
            np.put(self.feature_pheromone, f, updated_pheromone)
        
        time_pheromonesupdate_stop = time.time()
        self.time_pheromonesupdate = self.time_pheromonesupdate + (time_pheromonesupdate_stop - time_pheromonesupdate_start)
        
    def acoFS(self):
        """Compute the original ACO algorithm workflow. Firstly it resets the values of the ants (:py:meth:`featureselector.FeatureSelector.resetInitialValues`), 
        """

        self.defineLUT()
        for c in range(self.iterations):
            self.resetInitialValues()
            #print("Colony", c, ":")
            ia = 0
            for ia in range(self.number_ants):
                self.antBuildSubset(ia)
                #print("\tAnt", ia, ":")
                #print("\t\tPath:" ,self.ants[ia].feature_path)
                #print("\t\tCV-Acuraccy:", self.ant_accuracy[ia])
            self.updatePheromones()
            #print("\t\tPheromones:", self.feature_pheromone)

    def printTestingResults(self):
        """Function for printing the entire summary of the algorithm, including the test results.
        """
        print("The final subset of features is: ",self.ants[np.argmax(self.ant_accuracy)].feature_path)
        print("Number of features: ",len(self.ants[np.argmax(self.ant_accuracy)].feature_path))

        data_training_subset = self.data_training[:,self.ants[np.argmax(self.ant_accuracy)].feature_path]
        data_testing_subset = self.data_testing[:,self.ants[np.argmax(self.ant_accuracy)].feature_path]
                
        print("Subset of features dataset accuracy:")

        knn = KNeighborsClassifier()
        knn.fit(data_training_subset, self.class_training)
        knn_score = knn.score(data_testing_subset, self.class_testing) 
        print("\t CV-Training set: ", np.max(self.ant_accuracy))
        print("\t Testing set    : ", knn_score)

        print("\t Time elapsed reading data        : ", self.time_dataread)
        print("\t Time elapsed in LUT compute      : ", self.time_LUT)
        print("\t Time elapsed reseting values     : ", self.time_reset)
        print("\t Time elapsed in local search     : ", self.time_localsearch)
        print("\t Time elapsed updating pheromones : ", self.time_pheromonesupdate)
