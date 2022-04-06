# Pathfinder

Pathfinder is a innovative Feature Selection method based on the Ant Colony Optimization (ACO) algorithm.

Pathfinder is in continuous developing and improving, so the actual version is composed of a filter approach and a randomized search heuristic.

## Requirements

Pathfinder requires Python 3. It also depends on the following Python packages:

* [NumPy](https://numpy.org/).
* [Scikit-learn](https://scikit-learn.org/stable/).
* [Scipy](https://www.scipy.org/).
* [Sphinx](https://www.sphinx-doc.org/en/master/) if you want to generate documentation.

## Documentation

Pathfinder is fully documented in its [Github Pages](https://efficomp.github.io/Pathfinder/). In addition, in `doc` subfolder, the `Make` files contains a rule to generate [Sphinx](https://www.sphinx-doc.org/en/master/) documentation in the `doc/build/html` folder. 

## Usage

In `test` subfolder there is a `test.py` file that runs the application on a dataset. This program can work with two dataset versions:
 * A dataset divided in 4 files: training data, testing data, training classes, and testing classes. This files must be in MATLAB format (_.mat_). Usage:
    * `$ python test.py mat dataTrainingDataset classTrainingDataset dataTestingDataset classTestingDataset numberAnts numberColonies numberFeatures`

 * A dataset in only one file: all data together with the classes labels in the last column of the dataset. This file must be in comma separated format (_.csv_). Usage:
    * `$ python test.py csv dataset numberAnts numberColonies numberFeatures`

where:
- _.mat_ version:
  - dataTrainingDataset = Path to the _.mat_ file of data of the training dataset.
  - classTrainingDataset = Path to the _.mat_ file of corresponding classes of the trainin dataset.
  - dataTestingDataset = Path to the _.mat_ file of data of the testing dataset.
  - classTestingDataset = Path to the _.mat_ file of corresponding classes of the testing dataset.
- _.csv_ version: 
  - dataset = Path to the _.csv_ file of the entire dataset.
  - numberAnts = Number of ants for the algorithm.
  - numberColonies = Number of colonies for the algorithm.
  - numberFeatures = Number of features to  be selected.


## Publications

#### Conferences

1. A. Ortega, J.J. Escobar, J. Ortega, J. González, A. Alcayde, J. Munilla, and M. Damas. *Performance Study of Ant Colony Optimization for Feature Selection in EEG Classification*. In: **International Conference on Bioengineering and Biomedical Signal and Image Processing. BIOMESIP'2021**. Gran Canaria, Spain: Springer, July 2021, pp. 323-336. DOI: [10.1007/978-3-030-88163-4_28](https://doi.org/10.1007/978-3-030-88163-4_28)

2. A. Ortega, J.J. Escobar, M. Damas, A. Ortiz, and J. González. *Ant Colony Optimization for Feature Selection via a Filter-Randomized Search Heuristic*. In **Genetic and Evolutionary Computation Conference Companion. GECCO'2022**. Boston, MA, USA: ACM, July 2022. DOI: [10.1145/3520304.3528817](https://doi.org/10.1145/3520304.3528817)

## Funding

This work has been funded by:
* Spanish *[Ministerio de Ciencia, Innovación y Universidades](https://www.ciencia.gob.es/)* under grant number PGC2018-098813-B-C31. 
* [European Regional Development Fund (ERDF)](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: right">
  <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/miciu.jpg" height="60">
  <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/erdf.png" height="60">
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Pathfinder © 2021 [EFFICOMP](https://efficomp.ugr.es).
