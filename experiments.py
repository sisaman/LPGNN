import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile

import pandas as pd
import torch
from colorama import Fore, Style
from datasets import load_dataset
from tasks import LinkPrediction, NodeClassification, ErrorEstimation, Visualization
from argparse import ArgumentParser
torch.manual_seed(12345)


@torch.no_grad()
def transform_features(data, feature):
    if feature == 'deg':
        data = Data(**dict(data()))  # copy data to avoid changing the original
        num_nodes = data.num_nodes
        data.x = None
        data.num_nodes = num_nodes
        data = LocalDegreeProfile()(data)
    return data


def one_bit_response(data, eps):
    exp = math.exp(eps)
    delta = data.delta
    delta[delta == 0] = 1e-7  # avoid division by zero
    p = (data.x - data.alpha) / delta
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    x_priv = torch.bernoulli(p)
    data.x = data.priv_mask * x_priv + ~data.priv_mask * data.x
    return data


def privatize(data, pnr, pfr, eps):
    if pnr > 0 and pfr > 0:
        data = Data(**dict(data()))  # copy data to avoid changing the original
        mask = torch.zeros_like(data.x, dtype=torch.bool)
        n_rows = int(pnr * mask.size(0))
        n_cols = int(pfr * mask.size(1))
        priv_rows = torch.randperm(mask.size(0))[:n_rows]
        priv_cols = torch.randperm(mask.size(1))[:n_cols]
        mask[priv_rows.unsqueeze(1), priv_cols] = True
        data.priv_mask = mask
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        data.alpha = alpha
        data.delta = beta - alpha
        # noinspection PyTypeChecker
        data = one_bit_response(data, eps)
    return data


def visualize(dataset):
    eps_list = [0.1, 1, 10]
    for eps in eps_list:
        data = privatize(dataset, pnr=1, pfr=1, eps=eps)
        task = Visualization(data=data, model_name='gcn', epsilon=eps)
        result = task.run(max_epochs=500)
        df = pd.DataFrame(data=result['data'], columns=['x', 'y'])
        df['label'] = result['label']
        df.to_pickle(f'results/visualize_{data.name}_{eps}.pkl')


# noinspection PyShadowingNames
def experiment(args):
    task_class = {
        'nodeclass': NodeClassification,
        'linkpred': LinkPrediction,
        'errorest': ErrorEstimation,
        'visualize': Visualization
    }

    epsilons_methods = [0.1, 1, 3, 5, 7, 9]
    epsilons_priv_ratio = [1, 3, 5]
    epsilons_err_estimation = [.1, .2, .5, 1, 2, 5]
    private_node_ratios = [.2, .4, .6, .8, 1]
    private_feature_ratios = [.1, .2, .5, .75, 1]

    for task in args.tasks:
        task = task_class[task]

        for dataset_name in args.datasets:
            dataset = load_dataset(dataset_name, split_edges=(task.task_name in ['linkpred', 'visualize']))
            dataset = dataset.to('cuda')

            if task is Visualization:
                visualize(dataset)
                continue

            if task is ErrorEstimation: model_list = ['gcn']
            else: model_list = args.models

            for model in model_list:

                if task is ErrorEstimation: feature_list = ['priv']
                elif model == 'node2vec': feature_list = ['void']
                else: feature_list = args.features

                for feature in feature_list:
                    results = []

                    transformed_data = transform_features(dataset, feature)

                    if feature == 'priv': pnr_list = private_node_ratios
                    else: pnr_list = [0]  # when using other features / models

                    for pnr in pnr_list:

                        if feature != 'priv': pfr_list = [0]  # when no privacy
                        elif pnr == 1: pfr_list = private_feature_ratios  # vary pfr only when pnr = 1
                        else: pfr_list = [1]  # vary pnr only when pfr = 1

                        for pfr in pfr_list:

                            if task is ErrorEstimation: eps_list = epsilons_err_estimation
                            elif feature != 'priv': eps_list = [1]
                            elif pnr == pfr == 1: eps_list = epsilons_methods
                            else: eps_list = epsilons_priv_ratio

                            for eps in eps_list:

                                for run in range(args.repeats):

                                    print(
                                        Fore.BLUE +
                                        f'\ntask={task.task_name} / dataset={dataset_name} / model={model} / '
                                        f'feature={feature} / pnr={pnr} / pfr={pfr} / eps={eps} / run={run}'
                                        + Style.RESET_ALL
                                    )

                                    data = privatize(transformed_data, pnr=pnr, pfr=pfr, eps=eps)
                                    t = task(data=data, model_name=model, epsilon=eps, orig_features=dataset.x)
                                    result = t.run(max_epochs=args.epochs)

                                    if task is not ErrorEstimation:
                                        print(result)

                                    results.append((f'{model}+{feature}', pnr, pfr, eps, run, result))

                    df_result = pd.DataFrame(
                        data=results,
                        columns=['method', 'pnr', 'pfr', 'eps', 'run', 'perf']
                    )
                    df_result.to_pickle(f'results/{task.task_name}_{dataset_name}_{model}_{feature}.pkl')


if __name__ == '__main__':
    task_choices = ['nodeclass', 'linkpred', 'errorest', 'visualize']
    dataset_choices = ['cora', 'citeseer', 'pubmed', 'flickr', 'yelp']
    model_choices = ['gcn', 'node2vec', 'vgae']
    feature_choices = ['raw', 'priv', 'deg']
    parser = ArgumentParser()
    parser.add_argument('-t', '--tasks', nargs='*', choices=task_choices, default=task_choices)
    parser.add_argument('-d', '--datasets', nargs='*', choices=dataset_choices, default=dataset_choices)
    parser.add_argument('-m', '--models', nargs='*', choices=model_choices, default=model_choices)
    parser.add_argument('-f', '--features', nargs='*', choices=feature_choices, default=feature_choices)
    parser.add_argument('-r', '--repeats', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=500)

    args = parser.parse_args()
    print(args)
    experiment(args)
