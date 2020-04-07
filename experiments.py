import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
from colorama import Fore, Style
from datasets import load_dataset, EdgeSplit
from tasks import LinkPrediction, NodeClassification
from backup import ErrorEstimation


def experiment():
    tasks = [
        ErrorEstimation,
        # NodeClassification,
        # LinkPrediction,
    ]
    datasets = [
        'cora',
        # 'citeseer',
        # 'pubmed',
        # 'flickr'
    ]
    models = [
        'gcn',
        'node2vec'
    ]
    features = {
        'gcn': [
            'raw',
            'priv',
            'locd'
        ],
    }
    epsilons = [0.1, 1, 3, 5, 7, 9]
    epsilons_pr = [1, 3, 5]
    epsilons_err = [0.1, 0.2, 0.5, 1, 2, 5]
    private_ratios = [0.1, 0.2, 0.50, 1]
    repeats = 10

    for task in tasks:
        for dataset_name in datasets:
            dataset = load_dataset(dataset_name)
            model_list = ['gcn'] if task is ErrorEstimation else models
            for model in model_list:
                feature_list = ['priv'] if task is ErrorEstimation else features.get(model, ['void'])
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
                            for run in range(repeats):
                                print(Fore.BLUE + f'\ntask={task.task_name} / dataset={dataset_name} / model={model} / '
                                      f'feature={feature} / pr={pr} / eps={eps} / run={run}' + Style.RESET_ALL)
                                result = task(
                                    dataset=dataset, model_name=model, feature=feature, epsilon=eps,
                                    priv_dim=int(pr * dataset.num_node_features)
                                ).run()
                                if task is not ErrorEstimation:
                                    print(result)
                                results.append(
                                    (task.task_name, dataset_name, f'{model}+{feature}', pr, eps, run, result)
                                )

                    df_result = pd.DataFrame(
                        data=results,
                        columns=['task', 'dataset', 'method', 'pr', 'eps', 'run', 'perf']
                    )
                    df_result.to_pickle(f'results/{task.task_name}_{dataset_name}_{model}_{feature}.pkl')


def preprocess_datasets():
    print('---DATASET PREPROCESSING---')
    datasets = ['cora', 'citeseer', 'pubmed', 'flickr']
    for i, dataset in enumerate(datasets):
        torch.manual_seed(i)
        load_dataset(dataset, pre_transforms=[EdgeSplit()])
    print('---DONE---')


if __name__ == '__main__':
    # preprocess_datasets()
    torch.manual_seed(12345)
    experiment()
