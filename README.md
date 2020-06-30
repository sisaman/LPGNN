# When Differential Privacy Meets Graph Neural Networks

This repository is the official implementation of the paper [When Differential Privacy Meets Graph Neural Networks](https://arxiv.org/abs/2006.05535).  
By **Sina Sajadmanesh** and **Daniel Gatica-Perez**, Idiap Research Institute, EPFL. 


## Requirements

This code is implemented in Python 3.7, and requires the following packages to be installed:  
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.5.0
- [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning) >= 0.8.2
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.0.5
- [Numpy](https://numpy.org/install/) >= 1.18.5


## Usage

### Replicating the paper's results
In order to replicate our experiments and reproduce the paper's results, you must do the following steps:  
1. Run ``experiments.sh``. All the datasets will be downloaded automatically into ``datasets`` folder, and the results will be stored in ``results`` directory.
2. Go through ``results.ipynb`` notebook to visualize the results.

### Training and evaluating the paper's models
If you want to individually train and evaluate the models on any of the datasets mentioned in the paper, run the following command:  
```
python train.py [OPTIONS...]
```
Required arguments:  
```
-t, --task        <task name>               The graph learning task. Either "node" for node classification, or "link" for link prediction.
-d, --dataset     <dataset name>            The dataset to train on. One of "citeseer", "cora", "elliptic", "flickr", or "twitch".
-m, --methods     <list of methods>         The list of mechanisms to perturb node features. Can be "raw" to use original features, or local differentially private algorithms, including "pgc" for Private Graph Convolution, "pm" for Piecewise Mechanism, and "lm" for Laplace Mechanism.
-e, --eps         <list of epsilon values>  The list of epsilon values for LDP mechanisms. The values must be greater than zero. The "raw" method does not support this option.
```
Optional arguments:
```
-r, --repeats     <number of runs>          The number of times the experiment is repeated. Default is 10.
-o, --output-dir  <path>                    The path to store the results. Default is "./results".
    --device      <device>                  The device used for the training. Either "cpu" or "cuda". Default is "cuda".
```

### Measuring the estimation error

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```


## Results

Below is a summary of the performance of our differentially private GNN with different values of epsilon:


| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |



## Citation

If you find this code useful, please cite the following paper:  
```
@article{sajadmanesh2020differential,
  title={When Differential Privacy Meets Graph Neural Networks},
  author={Sajadmanesh, Sina and Gatica-Perez, Daniel},
  journal={arXiv preprint arXiv:2006.05535},
  year={2020}
}
```
