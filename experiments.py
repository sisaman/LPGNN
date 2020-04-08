import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
from colorama import Fore, Style
from datasets import load_dataset
from tasks import LinkPrediction, NodeClassification, ErrorEstimation
from argparse import ArgumentParser
torch.manual_seed(12345)


def experiment(args):
    task_class = {
        'nodeclass': NodeClassification,
        'linkpred': LinkPrediction,
        'errorest': ErrorEstimation
    }
    # datasets = [
    #     # 'cora',
    #     # 'citeseer',
    #     # 'pubmed',
    #     'flickr'
    # ]
    # models = [
    #     'gcn',
    #     'node2vec'
    # ]
    # features = {
    #     'gcn': [
    #         'raw',
    #         'priv',
    #         'locd'
    #     ],
    # }
    epsilons = [0.1, 1, 3, 5, 7, 9]
    epsilons_pr = [1, 3, 5]
    epsilons_err = [0.1, 0.2, 0.5, 1, 2, 5]
    private_ratios = [0.1, 0.2, 0.50, 1]

    for task in args.tasks:
        task = task_class[task]
        for dataset_name in args.datasets:
            dataset = load_dataset(dataset_name, task_name=task.task_name)
            model_list = ['gcn'] if task is ErrorEstimation else args.models
            for model in model_list:
                feature_list = ['priv'] if task is ErrorEstimation else args.features
                for feature in feature_list:
                    results = []
                    pr_list = private_ratios if feature == 'priv' else [0]
                    for pr in pr_list:

                        if task is ErrorEstimation:
                            eps_list = epsilons_err
                        elif feature == 'priv':
                            eps_list = epsilons if pr == 1 else epsilons_pr
                        else:
                            eps_list = [0]

                        for eps in eps_list:
                            for run in range(1 if task is ErrorEstimation else args.repeats):
                                t = task(
                                    dataset=dataset, model_name=model, feature=feature, epsilon=eps,
                                    priv_dim=int(pr * dataset.num_node_features)
                                )
                                print(Fore.BLUE + f'\ntask={task.task_name} / dataset={dataset_name} / model={model} / '
                                      f'feature={feature} / pr={pr} / eps={eps} / run={run}' + Style.RESET_ALL)
                                result = t.run(max_epochs=500)
                                if task is not ErrorEstimation:
                                    print(result)
                                results.append((f'{model}+{feature}', pr, eps, run, result))

                    df_result = pd.DataFrame(
                        data=results,
                        columns=['method', 'pr', 'eps', 'run', 'perf']
                    )
                    df_result.to_pickle(f'results/{task.task_name}_{dataset_name}_{model}_{feature}.pkl')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--tasks', nargs='+', choices=['nodeclass', 'linkpred', 'errorest'], required=True)
    parser.add_argument('-d', '--datasets', nargs='+', choices=['cora', 'citeseer', 'pubmed', 'flickr'], required=True)
    parser.add_argument('-m', '--models', nargs='*', choices=['gcn', 'node2vec'], default=['gcn'])
    parser.add_argument('-f', '--features', nargs='*', choices=['raw', 'priv', 'locd'], default=['raw'])
    parser.add_argument('-r', '--repeats', type=int, default=10)

    args = parser.parse_args()
    print(args)
    experiment(args)
