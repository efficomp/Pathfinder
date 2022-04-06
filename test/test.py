__author__ = 'Alberto Ortega'
__copyright__ = 'Pathfinder (c) 2021 EFFICOMP'
__credits__ =  'Spanish Ministerio de Ciencia, Innovacion y Universidades under grant number PGC2018-098813-B-C31. European Regional Development Fund (ERDF).'
__license__ = ' GPL-3.0'
__version__ = "2.0"
__maintainer__ = 'Alberto Ortega'
__email__ = 'aoruiz@ugr.es'

import sys
import os
sys.path.append(os.path.abspath('.'))
import time
from pathfinder.featureselector import FeatureSelector

if __name__ == "__main__":

    # TIME START
    start_time = time.time()

    # ACO CALL
    if sys.argv[1]=="mat":
        fs = FeatureSelector(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8])
    elif sys.argv[1]=="csv":
        fs = FeatureSelector(sys.argv[1], sys.argv[2], None ,int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    
    fs.acoFS()

    # TIME STOP
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    # PRINT TESTING RESULTS
    fs.printTestingResults()

    print("Elapsed time:")
    print("\t", elapsed_time, "seconds")
