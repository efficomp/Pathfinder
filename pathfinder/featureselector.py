__author__ = 'Alberto Ortega'
__copyright__ = 'Pathfinder (c) 2021 EFFICOMP'
__credits__ =  'Spanish Ministerio de Ciencia, Innovacion y Universidades under grant number PGC2018-098813-B-C31. European Regional Development Fund (ERDF).'
__license__ = ' GPL-3.0'
__maintainer__ = 'Alberto Ortega'
__email__ = 'aoruiz@ugr.es'

import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from pathfinder.ant import Ant
 
class FeatureSelector:
    """Class for Ant System Optimization algorithm designed for Feature Selection.
    
    :param algorithm: ACOFS, EACOFS or RACOFS, depending of which algorithm is going to be used.
    :param dataset_x: Training dataset.
    :param dataset_y: Class corresponding to the training dataset.
    :param predictions_x: Testing dataset.
    :param predictions_y: Class corresponding to the testing dataset.
    :param numberAnts: Number of ants of the colony.
    :param iterations: Number of colonies.
    :param alpha: Parameter which determines the weight of tau.
    :param beta: Parameter which determines the weight of eta.
    :param Q_constant: Parameter for the pheromones update function.
    :param initialPheromone: Initial value for the pheromones.
    :param evaporationRate: Rate of the pheromones evaporation.
    :type  algorithm: String
    :type  dataset_x: Numpy array
    :type  dataset_y: Numpy array
    :type  predictions_x: Numpy array
    :type  predictions_y: Numpy array
    :type  numberAnts: Integer
    :type  iterations: Integer
    :type  alpha: Float
    :type  beta: Float
    :type  Q_constant: Float
    :type  initialPheromone: Float
    :type  evaporationRate: Float
    """

    def __init__(self, algorithm, dataset_x, dataset_y, predictions_x, predictions_y, numberAnts, numberElitistAnts, iterations, alpha, beta, Q_constant, initialPheromone, evaporationRate):
        """Constructor method.
        """
        self.algorithm = algorithm
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.predictions_x = predictions_x
        self.predictions_y = predictions_y
        self.number_ants = numberAnts
        self.number_elitist_ants = numberElitistAnts
        self.ants = [Ant() for _ in range(self.number_ants)]
        self.number_features = len(self.dataset_x[0])
        self.iterations = iterations
        self.initial_pheromone = initialPheromone
        self.evaporation_rate = evaporationRate
        self.alpha = alpha
        self.beta = beta
        self.Q_constant = Q_constant
        self.feature_pheromone = np.full(self.number_features, self.initial_pheromone)
        self.unvisited_features = np.arange(self.number_features)
        self.ant_accuracy = np.zeros(self.number_ants)


    def resetInitialValues(self):
        """Initialize the ant array and assign each one a random initial feature.
        """

        self.ants = [Ant() for _ in range(self.number_ants)]
        initialFeaturesValues = np.arange(self.number_features)
        for i in range(self.number_ants):
            rand = random.choice(initialFeaturesValues)
            rand_index = np.where(initialFeaturesValues == rand)
            initialFeaturesValues = np.delete(initialFeaturesValues, rand_index)
            self.ants[i].feature_path.append(rand)

    def heuristicCost(self, index_ant, num_feature):
        """Defines the heuristic function for the local search and the cost of including the new feature in the ant subset of features.
        
        :param index_ant: Index of the ant array for the heuristic cost.
        :param num_feature: Index of the new feature to include.
        :type index_ant: Integer
        :type num_feature: Integer
        :return: Heuristic cost.
        :rtype: Float
        """
        
        # Define the new feature dataset subset of the ant and compute the accuracy
        new_features_list = np.array(self.ants[index_ant].feature_path)
        new_features_list = np.append(new_features_list, num_feature)
        new_subset = np.array(self.dataset_x[:,new_features_list])
        new_test_set = np.array(self.predictions_x[:,new_features_list])
        new_lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        new_lr_classifier.fit(new_subset, self.dataset_y)
        new_accuracy = new_lr_classifier.score(new_test_set, self.predictions_y)
        # Compare the old and new accuracy, if it's better it returns the diff but if isn't it returns 0
        if new_accuracy > self.ant_accuracy[index_ant]:
            h = new_accuracy - self.ant_accuracy[index_ant]
        else:
            h = 0
        return h

    def antBuildSubset(self, index_ant):
        """Global and local search for the ACO algorithm. It completes the subset of features of the ant searching
        between all the rest of features in order to find the best one which improves the subset of features of the 
        actual ant.    
        
        :param index_ant: Index of the ant array for the heuristic cost.
        :type index_ant: Integer
        :return: Ant with its subset of features completed.
        :rtype: :class:`ant`
        """

        # Initialize unvisited features and it removes the first of the ant actual subset
        self.unvisited_features = np.arange(self.number_features)
        indexes = np.where(np.in1d(self.unvisited_features, self.ants[index_ant].feature_path))[0]
        self.unvisited_features = np.delete(self.unvisited_features, indexes)

        run = True
        while run:
            # Initialize parameters
            p = np.zeros(np.size(self.unvisited_features))
            p_num = np.zeros(np.size(self.unvisited_features))
            p_den = 0
            eta = np.zeros(np.size(self.unvisited_features))
            tau = np.zeros(np.size(self.unvisited_features))

            # Define the actual feature dataset subset of the ant and compute the accuracy
            actual_features_list = self.ants[index_ant].feature_path
            actual_subset = np.array(self.dataset_x[:,actual_features_list])
            actual_test_set = np.array(self.predictions_x[:,actual_features_list])
            actual_lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
            actual_lr_classifier.fit(actual_subset, self.dataset_y)
            np.put(self.ant_accuracy, index_ant, actual_lr_classifier.score(actual_test_set, self.predictions_y))

            # Compute eta, tau and the numerator for each unvisited feature 
            for index_uf in range(len(self.unvisited_features)):
                np.put(eta,index_uf, self.heuristicCost(index_ant, self.unvisited_features[index_uf]))
                np.put(tau,index_uf, self.feature_pheromone[index_uf])
                np.put(p_num, index_uf, (tau[index_uf]**self.alpha) * (eta[index_uf]**self.beta))
            
            # Compute denominator as sumatory of the numerators, if it's 0 none of the rest of univisited features improves the actualsubset
            p_den = np.sum(p_num)
            if p_den == 0:
                run = False
            else:
                # Compute common probabilistic function of ACO
                for index_uf in range(len(self.unvisited_features)):
                    np.put(p, index_uf, p_num[index_uf] / p_den)    

                # Choose the feature with best probability and add to the ant subset
                index_best_p = np.argmax(p)
                self.ants[index_ant].feature_path.append(self.unvisited_features[index_best_p])
                # Remove the chosen feature of the unvisited features
                self.unvisited_features = np.delete(self.unvisited_features, index_best_p)

        print("\t\tPath:" ,self.ants[index_ant].feature_path)
        print("\t\tAcuraccy:", self.ant_accuracy[index_ant])
        #print("\t\tPheromones:", self.feature_pheromone[self.ants[index_ant].feature_path])


    def updatePheromones(self):
        """Update the pheromones trail depending on which variant of the algorithm it is selected.
        """
        if self.algorithm == 'ACOFS':
            for f in range(self.number_features):
                for ia in range(self.number_ants):
                    sum_delta = 0
                    if f in self.ants[ia].feature_path:
                        sum_delta += self.Q_constant / self.ant_accuracy[ia]

                updated_pheromone = ( 1 - self.evaporation_rate) * self.feature_pheromone[f] + sum_delta
                np.put(self.feature_pheromone, f, updated_pheromone)
        
        elif self.algorithm == 'EACOFS':  
            for f in range(self.number_features):
                for ia in range(self.number_ants):
                    sum_delta = 0
                    if f in self.ants[ia].feature_path:
                        sum_delta += self.Q_constant / self.ant_accuracy[ia]

                updated_pheromone = ( 1 - self.evaporation_rate) * self.feature_pheromone[f] + sum_delta
                np.put(self.feature_pheromone, f, updated_pheromone)

            index_best_accu = np.argmax(self.ant_accuracy)
            best_acc = np.amax(self.ant_accuracy)
            best_ant = self.ants[index_best_accu]
            for f in best_ant.feature_path:
                updated_pheromone_best_ant = self.feature_pheromone[f] + self.number_elitist_ants * self.Q_constant / best_acc
                np.put(self.feature_pheromone, f, updated_pheromone_best_ant)

        elif self.algorithm == 'RACOFS':
            aux_ants = np.arange(self.number_ants)
            sorted_index_ants = [x for _, x in sorted(zip(self.ant_accuracy, aux_ants))]
            
            for f in range(self.number_features):
                for iba in range(self.number_elitist_ants - 1):
                    sum_delta = 0
                    if f in self.ants[sorted_index_ants[iba]].feature_path:
                        sum_delta += (self.number_elitist_ants - iba) * self.Q_constant / self.ant_accuracy[sorted_index_ants[iba]]

                updated_pheromone = ( 1 - self.evaporation_rate) * self.feature_pheromone[f] + sum_delta
                np.put(self.feature_pheromone, f, updated_pheromone)

            index_best_accu = np.argmax(self.ant_accuracy)
            best_acc = np.amax(self.ant_accuracy)
            best_ant = self.ants[index_best_accu]
            for f in best_ant.feature_path:
                updated_pheromone_best_ant = self.feature_pheromone[f] + self.number_elitist_ants * self.Q_constant / best_acc
                np.put(self.feature_pheromone, f, updated_pheromone_best_ant)
    
    def buildFinalSolutions(self):
        """Compute the original ACO algorithm workflow. Firstly it resets the values of the ants (:py:meth:`featureselector.FeatureSelector.resetInitialValues`), 
        secondly it starts the local and global search of each ant (:py:meth:`featureselector.FeatureSelector.antBuildSubset`) of that colony, finally it updates
        the pheromones of the features (:py:meth:`featureselector.FeatureSelector.updatePheromones`). This procedure is repeated for each colony until
        complete the number of colonies.
        """

        for c in range(self.iterations):
            self.resetInitialValues()
            print("Colony", c, ":")
            ia = 0
            for ia in range(self.number_ants):
                print("\tAnt", ia, ":")
                self.antBuildSubset(ia)
            self.updatePheromones()

    def acoFS(self):
        """Call the :py:meth:`featureselector.FeatureSelector.buildFinalSolutions` and then it search the best ant of the last colony.
        :return: Path of the best ant of the last colony.
        :rtype: Array of Integers.
        """

        self.buildFinalSolutions()
        index_best_ant = np.argmax(self.ant_accuracy)

        return self.ants[index_best_ant].feature_path
