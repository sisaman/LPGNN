# When Differential Privacy Meets Graph Neural Networks

This repository is the official implementation of the paper [When Differential Privacy Meets Graph Neural Networks](https://arxiv.org/abs/2006.05535).  
By **Sina Sajadmanesh** and **Daniel Gatica-Perez**, Idiap Research Institute, EPFL. 


## Requirements

The following packages need to be installed:  

- [Python](https://www.python.org/downloads/) 3.7
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.5.0
- [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning) >= 0.8.2
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.0.5
- [Numpy](https://numpy.org/install/) >= 1.18.5


## Usage

### Replicating the paper's results

### Training and evaluating the paper's models

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
