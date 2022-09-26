# Deep Bayesian Optimization for Problems with High-Dimensional Structure

This library is the official repository for the paper [Deep Learning for Bayesian Optimization of Scientific Problems with High-Dimensional Structure]([https://arxiv.org/abs/2104.11667](https://openreview.net/pdf?id=tPMQ6Je2rB)). 
This branch contains additional comparison experiments. 
If you use any of this code, please cite:

```
@article{
kim2022deep,
title={Deep Learning for Bayesian Optimization of Scientific Problems with High-Dimensional Structure},
author={Samuel Kim and Peter Y Lu and Charlotte Loh and Jamie Smith and Jasper Snoek and Marin Solja{\v{c}}i{\'c}},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=tPMQ6Je2rB}
}
```

Note that this branch will only be minimally maintained, and will not be merged into the main branch.

## Installation

General requirements for all three problems (nanoparticle scattering, photonic crystal, chemistry) 
are listed in `requirements.txt`. 
Note that these requirements only cover the main results (e.g. BNNs, GP).

Some requirements are only needed for specific problems, namely:
* Photonic crystal (PC) - requirements are listed in `requirements_photonics.txt`.

* Chemistry - requirements are listed in `requirements_chem.txt`.

Note that Neural Tangents version 0.3.6 has changed the API and will not work, so it is recommended to install 0.3.5. 
For more details on installing Neural Tangents, its dependencies, or the GPU version, see here:
https://github.com/google/neural-tangents


## Experiments

### Bayesian Optimization of Composite Functions

The directory `BOCF/` is mostly copy + pasted from [this repository](https://github.com/RaulAstudillo/bocf) which appears to be 
an implementation of the paper [Bayesian Optimization of Composite Functions](https://arxiv.org/abs/1906.01537).
This repository performs Bayesian optimization of composite functions using multi-output GPs.

We have added the files `BOCF/opt_scatter_bocf.py` and `BOCF/opt_pc_bocf.py` for optimization on the nanoparticle
scattering and photonic crystal problems, respectively. Note that this library has its own set of requirements and
will require a separate conda environment from our main code.

### Interpretable Neural Architecture Search using Bayesian Optimisation with Weisfeiler-Lehman Kernel (NAS-BOWL)

The directory `nasbowl/` is mostly copy + pasted from the [official repository](https://github.com/xingchenwan/nasbowl) for the paper 
[Interpretable Neural Architecture Search via Bayesian Optimisation with Weisfeiler-Lehman Kernels](https://openreview.net/pdf?id=j9Rv7qdXjd).
This work uses WL kernels, which are designed to operate on graphs, to perform neural architecture search via Bayesian
optimization in which the architecture topology of a neural network is optimized. We adapt this code to perform
Bayesian optimization on the chemistry problem, which is labelled as "Graph-GP" in our paper.
The optimization can be run using the file `nasbowl/opt_chem_nasbowl.py`.

### Neural Tangents for Infinite Width Neural Networks

The files `opt_scatter_nt.py` and `opt_pc_nt.py` implement BO for the nanoparticle scattering and photonic crystal
tasks, respectively, using infinite width neural network approximations as GPs. The infinite neural networks are
implemented using the [Neural Tangents library](https://github.com/google/neural-tangents).

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

## License
We use the [MIT](https://choosealicense.com/licenses/mit/) for our code. 
Note that libraries we copy+pasted into here may have their own licenses.
