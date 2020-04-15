import warnings
from functools import partial

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile

import pandas as pd
import torch
from colorama import Fore, Style
from datasets import load_dataset, privatize
from tasks import LearningTask, ErrorEstimation, Visualization
from argparse import ArgumentParser
torch.manual_seed(12345)

priv_feature_set = {'bit', 'lap'}


def is_private(feature):
    return feature in priv_feature_set


@torch.no_grad()
def transform_features(data, feature):
    if feature == 'deg':
        data = Data(**dict(data()))  # copy data to avoid changing the original
        num_nodes = data.num_nodes
        data.x = None
        data.num_nodes = num_nodes
        data = LocalDegreeProfile()(data)
    return data


def visualize(dataset):
    eps_list = [0.1, 1, 5, 10]
    for eps in eps_list:
        print(
            Fore.BLUE +
            f'\ntask=visualize / dataset={dataset.name} / eps={eps}'
            + Style.RESET_ALL
        )
        data = privatize(dataset, pnr=1, pfr=1, eps=eps)
        task = Visualization(data=data, epsilon=eps)
        result = task.run()
        df = pd.DataFrame(data=result['data'], columns=['x', 'y'])
        df['label'] = result['label']
        df.to_pickle(f'results/visualize_{data.name}_{eps}.pkl')


# noinspection PyShadowingNames
def experiment(args):

    epsilons_methods = [0.1, 1, 3, 5, 7, 9]
    epsilons_priv_ratio = [1, 3, 5]
    epsilons_err_estimation = [.1, .2, .5, 1, 2, 5]

    for task in args.tasks:

        for dataset_name in args.datasets:
            dataset = load_dataset(dataset_name, split_edges=(task in ['linkpred', 'visualize']))
            dataset = dataset.to('cuda')

            if task == 'visualize':
                visualize(dataset)
                continue

            if task == 'errorest': model_list = ['gcn']
            else: model_list = args.models

            for model in model_list:

                if task == 'errorest': feature_list = priv_feature_set.intersection(args.features)
                elif model == 'node2vec': feature_list = ['void']
                else: feature_list = args.features

                for feature in feature_list:
                    results = []

                    transformed_data = transform_features(dataset, feature)

                    if is_private(feature): pnr_list = args.private_node_ratios
                    else: pnr_list = [0]  # when using other features / models

                    for pnr in pnr_list:

                        if not is_private(feature): pfr_list = [0]  # when no privacy
                        elif pnr == 1: pfr_list = args.private_feature_ratios  # vary pfr only when pnr = 1
                        else: pfr_list = [1]  # vary pnr only when pfr = 1

                        for pfr in pfr_list:

                            if task == 'errorest': eps_list = epsilons_err_estimation
                            elif not is_private(feature): eps_list = [1]
                            elif pnr == pfr == 1: eps_list = epsilons_methods
                            else: eps_list = epsilons_priv_ratio

                            for eps in eps_list:

                                task_class = {
                                    'nodeclass': partial(LearningTask, task_name='nodeclass'),
                                    'linkpred': partial(LearningTask, task_name='linkpred'),
                                    'errorest': partial(ErrorEstimation, orig_features=dataset.x),
                                    'visualize': Visualization
                                }

                                for run in range(args.repeats):

                                    print(
                                        Fore.BLUE +
                                        f'\ntask={task} / dataset={dataset_name} / model={model} / '
                                        f'feature={feature} / pnr={pnr} / pfr={pfr} / eps={eps} / run={run}'
                                        + Style.RESET_ALL
                                    )

                                    data = privatize(transformed_data, pnr=pnr, pfr=pfr, eps=eps, method=feature)
                                    t = task_class[task](
                                        data=data,
                                        model_name=model,
                                        epsilon=eps,
                                    )
                                    result = t.run()

                                    if task != 'errorest':
                                        print(result)

                                    results.append((f'{model}+{feature}', pnr, pfr, eps, run, result))

                    df_result = pd.DataFrame(
                        data=results,
                        columns=['method', 'pnr', 'pfr', 'eps', 'run', 'perf']
                    )

                    path = args.output
                    if path[-1] == '/': path = path[:-1]
                    df_result.to_pickle(f'{path}/{task}_{dataset_name}_{model}_{feature}.pkl')


if __name__ == '__main__':
    task_choices = ['nodeclass', 'linkpred', 'errorest', 'visualize']
    dataset_choices = ['cora', 'citeseer', 'pubmed', 'flickr', 'amazon-photo', 'amazon-computers', 'twitch',
                       'enzymes', 'proteins', 'frankenstein', 'bitcoin']
    model_choices = ['gcn', 'node2vec']
    feature_choices = ['raw', 'bit', 'deg', 'lap']
    private_node_ratios = [.2, .4, .6, .8, 1]
    private_feature_ratios = [.2, .4, .6, .8, 1]
    parser = ArgumentParser()
    parser.add_argument('-t', '--tasks', nargs='*', choices=task_choices, default=task_choices)
    parser.add_argument('-d', '--datasets', nargs='*', choices=dataset_choices, default=dataset_choices)
    parser.add_argument('-m', '--models', nargs='*', choices=model_choices, default=model_choices)
    parser.add_argument('-f', '--features', nargs='*', choices=feature_choices, default=feature_choices)
    parser.add_argument('-r', '--repeats', type=int, default=10)
    # parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument('--pnr', nargs='*', type=float, default=private_node_ratios, dest='private_node_ratios')
    parser.add_argument('--pfr', nargs='*', type=float, default=private_feature_ratios, dest='private_feature_ratios')

    args = parser.parse_args()
    print(args)
    experiment(args)
