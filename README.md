# Locally Private Graph Neural Networks

This repository is the official implementation of the paper [Locally Private Graph Neural Networks](https://arxiv.org/abs/2006.05535).  
By **Sina Sajadmanesh** and **Daniel Gatica-Perez**, Idiap Research Institute, EPFL. 


## Requirements

This code is implemented in Python 3.8, and relies on the following packages:  
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.6.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.6.1
- [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning) >= 1.0.2
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.1.3
- [Numpy](https://numpy.org/install/) >= 1.19.1
See requirements.txt for more details.


## Usage

### Replicating the paper's results
In order to replicate our experiments and reproduce the paper's results, you must do the following steps:  
1. Run ``python experiments.py``. All the datasets will be downloaded automatically into ``datasets`` folder, and the results will be stored in ``results`` directory.
2. Go through ``results.ipynb`` notebook to visualize the results.

### Training and evaluating individual models
If you want to individually train and evaluate the models on any of the datasets mentioned in the paper, run the following command:  
```
python train.py [OPTIONS...]
```
Required arguments:  
```
-d, --dataset   <string>    Dataset to train on. One of "citeseer", "cora", "pubmed", "facebook", "github", or "lastfm".
-m, --method    <string>    The method to perturb node features. Choices are "raw" (to use original features), "rnd" (for random features), "ohd" (for one-hot degree), or "mbm" for perturbation with multi-bit mechanism.
```
Optional arguments:
```
-e, --epsilons      <float sequence>    List of epsilon values to try for the the LDP mechanism. The values must be greater than zero. This is required if method is "mbm", and will be ignored for other non-private methods. Default is 0.
-k, --steps         <integer sequence>  List of KProp step parameters to try. Default is 1.
-l, --label-rates   <float sequence>    List of label rates to try as the fraction of training node (not the total nodes in the graph). Default is 1.0 (all the training nodes are used).
-r, --repeats       <integer>           Number of times the experiment is repeated. Default is 10.
-o, --output-dir    <path>              Path to store the results. Default is "./results".
    --device        <string>            Device used for the training. Either "cpu" or "cuda". Default is "cuda".
```
GNN Optional arguments:
```
--hidden-dim    <integer>   Dimension of the hidden layer of the GCN. Default is 16.
--dropout       <float>     Rate of dropout between zero and one. Default is 0.
--learning-rate <float>     Initial learning rate for the Adam optimizer. Default is 0.001.
--weight-decay  <float>     Weight decay (L2 penalty) for the Adam optimizer. Default is 0.
--aggregator    <string>    Neighborhood aggregator function. Either "gcn" or "mean". Default is "gcn".
```

The test result for each run will be saved as a csv file in ``<output-dir>`` directory.

#### Example
The followig command trains a vanilla GCN on the Cora dataset with one-hot degree features, an initial learning rate of 0.01, and a weight decay of 0.0001, and stores the test result in ``./temp`` folder:  
```
python train.py -d cora -m raw -o temp --learning-rate 0.01 --weight-decay 0.0001
```
The command below trains the LPGNN model on the Facebook dataset with a privacy budget of 1 and KProp step parameter 8:  
```
python train.py -d facebook -m mbm -e 1 -k 8
```

## Citation

If you find this code useful, please cite the following paper:  
```
@article{sajadmanesh2020differential,
  title={Locally Private Graph Neural Networks},
  author={Sajadmanesh, Sina and Gatica-Perez, Daniel},
  journal={arXiv preprint arXiv:2006.05535},
  year={2020}
}
```
