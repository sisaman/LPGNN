# Locally Private Graph Neural Networks

This repository is the official implementation of the paper [Locally Private Graph Neural Networks](https://arxiv.org/abs/2006.05535) (ACM CCS '21).  

**Abstract**

> 	Graph Neural Networks (GNNs) have demonstrated superior performance in learning node representations for various graph inference tasks. However, learning over graph data can raise privacy concerns when nodes represent people or human-related variables that involve sensitive or personal information. In this paper, we study the problem of node data privacy, where graph nodes (e.g., social network users) have potentially sensitive data that is kept private, but they could be beneficial for a central server for training a GNN over the graph. To address this problem, we propose a privacy-preserving, architecture-agnostic GNN learning framework with formal privacy guarantees based on Local Differential Privacy (LDP). Specifically, we develop a locally private mechanism to perturb and compress node features, which the server can efficiently collect to approximate the GNN's neighborhood aggregation step. Furthermore, to improve the accuracy of the estimation, we prepend to the GNN a denoising layer, called KProp, which is based on the multi-hop aggregation of node features. Finally, we propose a robust algorithm for learning with privatized noisy labels, where we again benefit from KProp's denoising capability to increase the accuracy of label inference for node classification. Extensive experiments conducted over real-world datasets demonstrate that our method can maintain a satisfying level of accuracy with low privacy loss.  

![figure](https://raw.githubusercontent.com/sisaman/LPGNN/master/LPGNN.png)


## Requirements

This code is implemented in Python 3.9, and relies on the following packages:  
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.8.1
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.7.0
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.2.4
- [Numpy](https://numpy.org/install/) >= 1.20.2
- [Seaborn](https://seaborn.pydata.org/) >= 0.11.1  

#### Note: For the DGL-based implementation, switch to the [DGL branch](https://github.com/sisaman/LPGNN/tree/DGL).

## Usage

### Replicating the paper's results
In order to replicate our experiments and reproduce the paper's results, you must do the following steps:  
1. Run ``python experiments.py -n LPGNN create --LPGNN --baselines``
2. Run ``python experiments.py -n LPGNN exec --all``  
   All the datasets will be downloaded automatically into ``datasets`` folder, and the results will be stored in ``results`` directory.
2. Go through ``results.ipynb`` notebook to visualize the results.

### Training individual models
If you want to individually train and evaluate the models on any of the datasets mentioned in the paper, run the following command:  
```
python main.py [OPTIONS...]

dataset arguments:
  -d              <string>       name of the dataset (choices: cora, pubmed, facebook, lastfm) (default: cora)
  --data-dir      <path>         directory to store the dataset (default: ./datasets)
  --data-range    <float pair>   min and max feature value (default: (0, 1))
  --val-ratio     <float>        fraction of nodes used for validation (default: 0.25)
  --test-ratio    <float>        fraction of nodes used for test (default: 0.25)

data transformation arguments:
  -f              <string>       feature transformation method (choices: raw, rnd, one, ohd) (default: raw)
  -m              <string>       feature perturbation mechanism (choices: mbm, 1bm, lpm, agm) (default: mbm)
  -ex             <float>        privacy budget for feature perturbation (default: inf)
  -ey             <float>        privacy budget for label perturbation (default: inf)

model arguments:
  --model         <string>       backbone GNN model (choices: gcn, sage, gat) (default: sage)
  --hidden-dim    <integer>      dimension of the hidden layers (default: 16)
  --dropout       <float>        dropout rate (between zero and one) (default: 0.0)
  -kx             <integer>      KProp step parameter for features (default: 0)
  -ky             <integer>      KProp step parameter for labels (default: 0)
  --forward       <boolean>      applies forward loss correction (default: True)

trainer arguments:
  --optimizer     <string>       optimization algorithm (choices: sgd, adam) (default: adam)
  --max-epochs    <integer>      maximum number of training epochs (default: 500)
  --learning-rate <float>        learning rate (default: 0.01)
  --weight-decay  <float>        weight decay (L2 penalty) (default: 0.0)
  --patience      <integer>      early-stopping patience window size (default: 0)
  --device        <string>       desired device for training (choices: cuda, cpu) (default: cuda)

experiment arguments:
  -s              <integer>      initial random seed (default: None)
  -r              <integer>      number of times the experiment is repeated (default: 1)
  -o              <path>         directory to store the results (default: ./output)
  --log           <boolean>      enable wandb logging (default: False)
  --log-mode      <string>       wandb logging mode (choices: individual,collective) (default: individual)
  --project-name  <string>       wandb project name (default: LPGNN)
```

The test result for each run will be saved as a csv file in the directory specified by  
``-o`` option (default: ./output).

## Citation

If you find this code useful, please cite the following paper:  
```bibtex
@inproceedings{sajadmanesh2020differential,
  title={Locally Private Graph Neural Networks},
  author={Sajadmanesh, Sina and Gatica-Perez, Daniel},
  booktitle={Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS '21)},
  year={2021}
}
```
