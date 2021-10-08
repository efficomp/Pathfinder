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
In ``test`` subfolder there is a ``test.py`` that runs the application on a dataset. The usage is:

.. code-block:: shell

    $ python test.py version dataTrainingDataset classTrainingDataset dataTestingDataset classTestingDataset numberAnts numberElitistAnts numberColonies

where:
        
    - ``version`` = ACOFS, EACOFS or RACOFS.
    - ``dataTrainingDataset`` = Path to the file of data of the training dataset.
    - ``classTrainingDataset`` = Path to the file of corresponding classes of the training dataset.
    - ``dataTestingDataset`` = Path to the file of data of the testing dataset.
    - ``classTestingDataset`` = Path to the file of corresponding classes of the testing dataset.
    - ``numberAnts`` = Number of ants for the algorithm.
    - ``numberElitistAnts`` = Number of elitist ants for the algorithm. In case of choosing ACOFS version, this argument does not apply.
    - ``numberColonies`` = Number of colonies for the algorithm.