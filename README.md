# Root-Join-Experiments

This repository is based on the [sisap25-example-python repository](https://github.com/sisap-challenges/sisap25-example-python) and contains a Python implementation of the Root Join algorithm (as described in this [paper](https://www.sciencedirect.com/science/article/pii/S0306437920300211?via%3Dihub)). It integrates various techniques without constraints.


## Step for running

A working installation of Python 3.12 is required, along with the following libraries: Numpy, H5py, Scikit-learn, memory-profiler and Pandas.

The steps are the following:

1. Clone the repository
2. Install requirements
3. Run
4. Evaluate

## Clone the repository

```bash
git clone https://github.com/Fa11ingDeep/Root-Join-Experiments.git
cd Root-Join-Experiments
```
## Install the requirements

```bash
pip install -r requirements.txt 
```
## Run

Run the second task on an example input (by default is set with gooaq dataset)
It will automatically take care of downloading the necessary example dataset.

```bash
python root_join.py
```
or

```bash
python root_join.py --task task2 --dataset gooaq
```
## Evaluation

will produce a summary file of the results with the computed recall against the ground truth data.

```bash
python eval.py results.csv
```
