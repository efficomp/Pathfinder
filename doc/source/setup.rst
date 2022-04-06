Setup
=====
In this section it is explained how to use Pathfinder.

Installation
************
It is necessary to install ``numpy``, ``tensorflow``, ``scikit-learn``, and ``scipy`` python packages to run the program ``test.py``.
These installations can be done with ``pip -r requirements.txt``.

Usage
*****
This method needs a dataset divided in training dataset and a testing dataset, each one divided in data and their corresponding classes. Datasets must be on MATLAB files.
MATLAB variables for data must be named as "data" and MATLAB variables for corresponding classes must be named as "class".

In ``test`` subfolder there is a ``test.py`` file that runs the application on a dataset. This program can work with two dataset versions:
 * A dataset divided in 4 files: training data, testing data, training classes, and testing classes. This files must be in MATLAB format (.mat). Usage:
 
 .. code-block:: shell

    $ python test.py mat dataTrainingDataset classTrainingDataset dataTestingDataset classTestingDataset numberAnts numberColonies numberFeatures
 
 * A dataset in only one file: all data together with the classes labels in the last column of the dataset. This file must be in comma separated format (.csv). Usage:

 .. code-block:: shell

    $ python test.py csv dataset numberAnts numberColonies numberFeatures

Where:

``.mat`` version:
  - ``dataTrainingDataset`` = Path to the .mat file of data of the training dataset.
  - ``classTrainingDataset`` = Path to the .mat file of corresponding classes of the trainin dataset.
  - ``dataTestingDataset`` = Path to the .mat file of data of the testing dataset.
  - ``classTestingDataset`` = Path to the .mat file of corresponding classes of the testing dataset.

``.csv`` version:
  - ``dataset`` = Path to the .csv file of the entire dataset.
  - ``numberAnts`` = Number of ants for the algorithm.
  - ``numberColonies`` = Number of colonies for the algorithm.
  - ``numberFeatures`` = Number of features to  be selected.    