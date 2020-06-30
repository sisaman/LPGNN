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

### Training and evaluating individual models
If you want to individually train and evaluate the models on any of the datasets mentioned in the paper, run the following command:  
```
python train.py [OPTIONS...]
```
Required arguments:  
```
-t, --task      <string>            Graph learning task. Either "node" for node classification, or "link" for link prediction.
-d, --dataset   <string>            Dataset to train on. One of "citeseer", "cora", "elliptic", "flickr", or "twitch".
-m, --methods   <string sequence>   List of mechanisms to perturb node features. Can be "raw" to use original features, or local differentially private algorithms, including "pgc" for Private Graph Convolution, "pm" for Piecewise Mechanism, and "lm" for Laplace Mechanism.
-e, --eps       <float sequence>    List of epsilon values for LDP mechanisms. The values must be greater than zero. The "raw" method does not support this option.
```
Optional arguments:
```
-r, --repeats       <integer>   Number of times the experiment is repeated. Default is 10.
-o, --output-dir    <path>      Path to store the results. Default is "./results".
    --device        <string>    Device used for the training. Either "cpu" or "cuda". Default is "cuda".
```
Optional arguments for node classification (with ``python train.py --task node``)
```
--hidden-dim    <integer>   Dimension of the hidden layer of the GCN. Default is 16.
--dropout       <float>     Rate of dropout between zero and one. Default is 0.5.
--learning-rate <float>     Initial learning rate for the Adam optimizer. Default is 0.001.
--weight-decay  <float>     Weight decay (L2 penalty) for the Adam optimizer. Default is 0.
--min-epochs    <integer>   Minimum number of training epochs. Default is 10.
--max-epochs    <integer>   Maximum number of training epochs. Default is 500.
--min-delta     <float>     Minimum change in the validation loss to qualify as an improvement in the early stopping, i.e. an absolute change of less than min-delta, will count as no improvement. Default is 0.
--patience      <integer>   Number of validation epochs with no improvement after which training will be stopped. Default is 20.
```
Optional arguments for link prediction (with ``python train.py --task link``)
```
--encoder-hidden-dim        <integer>   Dimension of the hidden layer of the VGAE encoder. Default is 32.
--encoder-output-dim        <integer>   Dimension of the output layer of the VGAE encoder. Default is 16.
--dropout                   <float>     Rate of dropout between zero and one. Default is 0.
--learning-rate             <float>     Initial learning rate for the Adam optimizer. Default is 0.001.
--weight-decay              <float>     Weight decay (L2 penalty) for the Adam optimizer. Default is 0.
--min-epochs                <integer>   Minimum number of training epochs. Default is 100.
--max-epochs                <integer>   Maximum number of training epochs. Default is 500.
--min-delta                 <float>     Minimum change in the validation loss to qualify as an improvement in the early stopping, i.e. an absolute change of less than min-delta, will count as no improvement. Default is 0.
--patience                  <integer>   Number of validation epochs with no improvement after which training will be stopped. Default is 10.
--check-val-every-n-epoch   <integer>   Evaluate validation loss every n train epochs. Default is 10.
```
The test result for each method will be saved as a pickled Pandas dataframe in ``<output-dir>`` directory under the name `<task>_<dataset>_<method>.pkl`.

#### Example
The followig command trains a GCN for node classification on Cora dataset with raw features, an initial learning rate of 0.01, and a weight decay of 0.0001, and stores the test result in ``./temp`` folder:  
```
python train.py -t node -d cora -m raw -o temp --learning-rate 0.01 --weight-decay 0.0001
```
The command below trains different VGAE models with an encoder output dimension of 32 for link prediction on Twitch dataset for each PGC and PM feature perturbation methods and for each epsilon value of 1, 5, and 9 using the device "cpu":  
```
python train.py -t link -d twitch -m pgc pm -e 1 5 9 --device cpu ---encoder-output-dim 32
```

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
