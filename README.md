# Deep Bayesian Optimization for Problems with High-Dimensional Structure

This library is the official repository for the paper [Deep Learning for Bayesian Optimization of Scientific Problems with High-Dimensional Structure](https://arxiv.org/abs/2104.11667). 
If you use any of this code, please cite:

```
@article{kim2021deep,
  title={Deep Learning for Bayesian Optimization of Scientific Problems with High-Dimensional Structure},
  author={Kim, Samuel and Lu, Peter Y and Loh, Charlotte and Smith, Jamie and Snoek, Jasper and Solja{\v{c}}i{\'c}, Marin},
  journal=arXiv preprint arXiv:2104.11667},
  pages={arXiv--2104},
  year={2021}
}
```

We implement Bayesian Optimization (BO) using Bayesian neural networks (BNNs) as the surrogate model. 
Our library allows BNNs to efficiently take advantage of auxiliary information, 
which is intermediate or additional information provided by the black-box function from which the objective can be computed cheaply.
We also enable BNNs to operate on high-dimensional input spaces such as images and graphs.

## Installation

General requirements for all three problems (nanoparticle scattering, photonic crystal, chemistry) 
are listed in `requirements.txt`. 
Note that these requirements only cover the main results (e.g. BNNs, GP).

Some requirements are only needed for specific problems, namely:
* Photonic crystal (PC) - requirements are listed in `requirements_photonics.txt`.

* Chemistry - requirements are listed in `requirements_chem.txt`.

## Usage

An example of using BO with auxiliary information on the nanoparticle scattering problem is given below.

```
python opt_scatter.py --opt nn2 --uncertainty ensemble --acquisition EI --objective hipass --n-units 256 --n-layers 8
```

The algorithm used is controlled by the `--opt` argument. `nn` denotes using a BNN (excluding infinite-width networks)
and `nn2` marks using BNNs with auxiliary information. 
The exact BNN architecture is specified by `--uncertainty` 
with choices and hyperparameter arguments found in  `lib/models/nn`

The size of X_pool for the photonic crystal task is m=10^4 by default due to increased memory requirements. If you want
to use m=10^5 similar to the paper, then you can use the command 

```
python opt_pc.py --opt cnn2 --uncertainty ensemble --af-m 100000
```

For applying GPs (or any other optimization algorithm that operates over continuous spaces) to the chemistry problem,
run
```python chem/soap.py```
first to generate the SOAP descriptors of the QM9 dataset.

## Code Organization

Main optimization files are in the root directory, `opt_scatter.py`, `opt_pc.py`, and `opt_chem.py`.

### Data Generation

* `scattering/` contains files for generating the nanoparticle scattering dataset.
* `pc/` contains files for generating the photonic crystal dataset.
* The `spektral/` directory is copied over from the TensorFlow 1 version of the Spektral library 
https://github.com/danielegrattarola/spektral/tree/tf1. 
(Note that the most recent version of Spektral uses TensorFlow 2.)

### BNN Models

BNN models are located in the directory `lib/models/`. 
* `lib/models/nn.py`: used to set up ArgParse arguments and select the appropriate BNN model.
* `lib/models/nn_base.py`: contains base class, dropout, and Neural Linear for fully-connected and convolutional NNs. 
* `lib/models/bbb.py`: contains Bayes by Backprop (BBB) AKA mean-field approximation for fully-connected and convolutional NNs. 
* `lib/models/ensembles.py`: contains ensembles of fully-connected and convolutional NNs.
* `lib/models/gnn.py`: All variants of graph neural network (GNN) models. Due to the additional dependencies, we have
separated out all GNN variants used for the chemistry task into this file, which includes ensembles, BBB, and neural linear variants of GNNs.

## Additional Experiments

For conciseness and ease of use, we only include select methods (most of the BNNs, GPs, random sampling) in this library.
If you would like to look at the additional experiments studied in our paper (GP variants, additional BNNs, non-BO methods),
please check out the `fullexperiments` branch of this repository: https://github.com/samuelkim314/DeepBO/tree/fullexperiments

## License
[MIT](https://choosealicense.com/licenses/mit/)
