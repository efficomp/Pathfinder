__author__ = 'Alberto Ortega'
__copyright__ = 'Pathfinder (c) 2021 EFFICOMP'
__credits__ =  'Spanish Ministerio de Ciencia, Innovacion y Universidades under grant number PGC2018-098813-B-C31. European Regional Development Fund (ERDF).'
__license__ = ' GPL-3.0'
__version__ = "2.0"
__maintainer__ = 'Alberto Ortega'
__email__ = 'aoruiz@ugr.es'

class Ant:
    """
    Class that represents an ant and its path followed through
    the dataset features.

    :param feature_path: Set of features travelled by the ant.
    :type feature_path: Array of Integers
    """

    def __init__(self):
        """Constructor method
        """

        self.feature_path = [] 