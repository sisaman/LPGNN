import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
from tqdm import tqdm, trange

from datasets import load_dataset, EdgeSplit
from tasks import LinkPrediction, NodeClassification, ErrorEstimation

torch.manual_seed(12345)


def experiment():
    tasks = [
        NodeClassification,
        LinkPrediction,
        ErrorEstimation
    ]
    datasets = [
        'cora',
        'citeseer',
        'pubmed',
        'flickr'
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
    repeats = 1

    for task in tqdm(tasks, desc='task'):
        for dataset_name in tqdm(datasets, desc=f'(task={task.task_name}) dataset', leave=False):
            transform = EdgeSplit() if task is LinkPrediction else None
            dataset = load_dataset(dataset_name, transform=transform)
            model_list = ['gcn'] if task is ErrorEstimation else models
            for model in tqdm(model_list, desc=f'(dataset={dataset_name}) model', leave=False):
                feature_list = ['priv'] if task is ErrorEstimation else features.get(model, ['void'])
                for feature in tqdm(feature_list, desc=f'(model={model}) feature', leave=False):
                    results = []
                    pr_list = private_ratios if feature == 'priv' else [0]
                    for pr in tqdm(pr_list, desc=f'(feature={feature}) private ratio', leave=False):

                        if task is ErrorEstimation:
                            eps_list = epsilons_err
                        elif feature == 'priv':
                            eps_list = epsilons if pr == 1 else epsilons_pr
                        else:
                            eps_list = [0]

                        for eps in tqdm(eps_list, desc=f'(ratio={pr}) epsilon', leave=False):
                            task_instance = task(
                                dataset, model, feature, eps,
                                priv_dim=int(pr * dataset.num_node_features)
                            )
                            for run in trange(repeats, desc=f'(epsilon={eps}) run', leave=False):
                                result = task_instance.run(max_epochs=500)
                                results.append((f'{model}+{feature}', pr, eps, run, result))

                    df_result = pd.DataFrame(data=results, columns=['method', 'pr', 'eps', 'run', 'perf'])
                    df_result.to_pickle(f'results/{task.task_name}_{dataset_name}_{model}_{feature}.pkl')


if __name__ == '__main__':
    experiment()
