# Locally Private Graph Neural Networks

This repository is the DGL-based implementation of the paper [Locally Private Graph Neural Networks](https://arxiv.org/abs/2006.05535).  


## Requirements

The environment is managed with [uv](https://docs.astral.sh/uv/) and supports Python 3.9–3.11 on Linux x86_64. It uses [PyTorch 2.0.1 with CUDA 11.8](https://pytorch.org/get-started/previous-versions/#v201) and [DGL 1.1.3 with CUDA 11.8](https://data.dgl.ai/wheels/cu118/repo.html). A working NVIDIA driver is required for GPU training.

Install the dependencies and run training with:

```sh
uv sync --python 3.9
```

`uv sync` also installs the `lpgnn_dgl` package from `src/lpgnn_dgl` in editable mode. The root-level `train.py` and `experiments.py` remain the CLI scripts. Prefix the Python commands below with `uv run` to use this environment, for example `uv run python train.py --help`.

#### Note: For the Pytorch-Geometric implementation, switch to the [main branch](https://github.com/sisaman/LPGNN).

**Disclaimer:** The results in the paper have been obtained using the pytorch-geometric implementation. The DGL version has not been fully tested and thus might not work as expected or there might be some slight discrepancies between the paper and the results achieved using this code.

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
python train.py [OPTIONS...]

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
```
@article{sajadmanesh2020differential,
  title={Locally Private Graph Neural Networks},
  author={Sajadmanesh, Sina and Gatica-Perez, Daniel},
  journal={arXiv preprint arXiv:2006.05535},
  year={2020}
}
```
