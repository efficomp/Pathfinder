# Pathfinder

Pathfinder is a innovative Feature Selection method based on the Ant Colony Optimization (ACO) algorithm.

It has been developed with 2 additional variants: Elitist variant and Ranked-Based variant. This method needs a dataset divided in training dataset and a testing dataset, each one divided in data and their corresponding classes.

## Requirements

Pathfinder requires Python 3. It also depends on the following Python packages:

* [NumPy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Scipy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Sphinx](https://www.sphinx-doc.org/en/master/)

## Documentation

Pathfinder is fully documented in its [Github Pages](https://efficomp.github.io/Pathfinder/). In addition, in `doc` subfolder, the `Make` files contains a rule to generate [Sphinx](https://www.sphinx-doc.org/en/master/) documentation in the `doc/build/html` folder. 

## Usage

This program needs a dataset divided in training dataset and a testing dataset and each one divided in data and their corresponding classes.
In `test` subfolder there is a `test.py` file that runs the application on a dataset. The usage is:

`$ python test.py version dataTrainingDataset classTrainingDataset dataTestingDataset classTestingDataset numberAnts numberElitistAnts numberColonies`

where:
- version = ACOFS, EACOFS (Elitist), or RACOFS (Ranked-Based).
- dataTrainingDataset = Path to the file of data of the training dataset.
- classTrainingDataset = Path to the file of corresponding classes of the trainin dataset.
- dataTestingDataset = Path to the file of data of the testing dataset.
- classTestingDataset = Path to the file of corresponding classes of the testing dataset.
- numberAnts = Number of ants for the algorithm.
- numberElitistAnts = Number of elitist ants for the algorithm. In case of choosing ACOFS for version, this argument should be 0.
- numberColonies = Number of colonies for the algorithm.

## Publications

* A. Ortega, J.J. Escobar, J. Ortega, J. González, A. Alcayde, J. Munilla and M. Damas. *Performance Study of Ant Colony Optimization for Feature Selection in EEG Classification*. In: **International Conference on Bioengineering and Biomedical Signal and Image Processing. BIOMESIP 2021**, pp. 323-336. https://doi.org/10.1007/978-3-030-88163-4_28

## Acknowledgments

This work was supported by project *New Computing Paradigms and Heterogeneous Parallel Architectures for High-Performance and Energy Efficiency of Classification and Optimization Tasks on Biomedical Engineering Applications* ([HPEE-COBE] (https://efficomp.ugr.es/research/projects/hpee-cobe/)), with reference PGC2018-098813-B-C31, funded by the Spanish *[Ministerio de Ciencia, Innovación y Universidades](https://www.ciencia.gob.es/)*, and by the [European Regional Development Fund (ERDF)](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: right">
  <a href="https://www.ciencia.gob.es/">
    <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/miciu.jpg" height="70">
  </a>
  <a href="https://ec.europa.eu/regional_policy/en/funding/erdf/">
    <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/erdf.png" height="70">
  </a>
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Pathfinder © 2021 [EFFICOMP](https://efficomp.ugr.es).
