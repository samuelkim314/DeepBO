# Deep Bayesian Optimization for Problems with High-Dimensional Structure

This library is the official repository for the following paper. If you use any of this code, please cite:

Kim, S., Lu, P. Y., Loh, C., Smith, J., Snoek, J., & Soljačić, M. (2021). Deep Learning for Bayesian Optimization of Scientific Problems with High-Dimensional Structure. *arXiv preprint arXiv:2104.11667.*

This library implements Bayesian Optimization (BO) using Bayesian neural networks (BNNs) as the surrogate model. 
Our library allows BNNs to efficiently take advantage of auxiliary information, which is intermediate or additional information provided by the black-box function from which the objective can be computed cheaply.

## Installation

Requirements are listed in `requirements.txt`. Note that these requirements cover all optimization algorithms
(e.g. GP, BNNs, LIPO, DIRECT-L) and all three problems (nanoparticle, photonic crystal, chemistry).
Some requirements are only needed for specific problems, namely:

Photonic crystal:

```
pymeep
```

Chemistry:

```
keras
networkx
requests
pandas
ase
dscribe
```

Note that Neural Tangents version 0.3.6 has changed the API and will not work, so it is recommended to install 0.3.5. 
For more details on installing Neural Tangents, its dependencies, or the GPU version, see here:
https://github.com/google/neural-tangents

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

## License
[MIT](https://choosealicense.com/licenses/mit/)
