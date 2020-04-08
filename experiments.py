import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
from colorama import Fore, Style
from datasets import load_dataset
from tasks import LinkPrediction, NodeClassification, ErrorEstimation
from argparse import ArgumentParser
torch.manual_seed(12345)


# noinspection PyShadowingNames
def experiment(args):
    task_class = {
        'nodeclass': NodeClassification,
        'linkpred': LinkPrediction,
        'errorest': ErrorEstimation
    }

    epsilons = [0.1, 1, 3, 5, 7, 9]
    epsilons_pr = [1, 3, 5]
    epsilons_err = [0.1, 0.2, 0.5, 1, 2, 5]
    private_ratios = [0.1, 0.2, 0.50, 1]

    for task in args.tasks:
        task = task_class[task]

        for dataset_name in args.datasets:
            data = load_dataset(dataset_name, task_name=task.task_name)
            data = data.to('cuda')

            if task is ErrorEstimation: model_list = ['gcn']
            else: model_list = args.models

            for model in model_list:

                if task is ErrorEstimation: feature_list = ['priv']
                elif model == 'node2vec': feature_list = ['void']
                else: feature_list = args.features

                for feature in feature_list:
                    results = []

                    if feature == 'priv': pr_list = private_ratios
                    else: pr_list = [0]

                    for pr in pr_list:

                        if task is ErrorEstimation: eps_list = epsilons_err
                        elif feature != 'priv': eps_list = [0]
                        elif pr == 1: eps_list = epsilons
                        else: eps_list = epsilons_pr

                        for eps in eps_list:
                            for run in range(1 if task is ErrorEstimation else args.repeats):
                                t = task(
                                    data=data, model_name=model, feature=feature, epsilon=eps,
                                    priv_dim=int(pr * data.num_node_features)
                                )
                                print(Fore.BLUE + f'\ntask={task.task_name} / dataset={dataset_name} / model={model} / '
                                      f'feature={feature} / pr={pr} / eps={eps} / run={run}' + Style.RESET_ALL)
                                result = t.run(max_epochs=args.epochs)
                                if task is not ErrorEstimation:
                                    print(result)
                                results.append((f'{model}+{feature}', pr, eps, run, result))

                    df_result = pd.DataFrame(
                        data=results,
                        columns=['method', 'pr', 'eps', 'run', 'perf']
                    )
                    df_result.to_pickle(f'results/{task.task_name}_{dataset_name}_{model}_{feature}.pkl')


if __name__ == '__main__':
    task_choices = ['nodeclass', 'linkpred', 'errorest']
    dataset_choices = ['cora', 'citeseer', 'pubmed', 'flickr']
    model_choices = ['gcn', 'node2vec']
    feature_choices = ['raw', 'priv', 'locd']
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
