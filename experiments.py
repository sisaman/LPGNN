import time
from loggers import PandasLogger
from argparse import ArgumentParser
from colorama import Fore, Style
from datasets import load_dataset, get_available_datasets
from mechanisms import privatize, available_mechanisms
from tasks import LearningTask, ErrorEstimation, Task
from pytorch_lightning import seed_everything


def get_arg(value, default):
    return default if value is None else value


def generate_config(feature):
    if feature == 'raw':
        return [{
            'eps': 0,
            'rfr': rfr,
            'pfr': 0
        } for rfr in get_arg(args.rfr_list, [0.2, 0.4, 0.6, 0.8, 1])]

    configs = [{
        'eps': eps,
        'rfr': 0,
        'pfr': 1
    } for eps in get_arg(args.eps_list, [1, 3, 5, 7, 9])]

    if feature == 'bit':
        configs += [{
            'eps': eps,
            'rfr': 0.2,
            'pfr': pfr
        } for eps in get_arg(args.eps_list, [3, 5, 7]) for pfr in get_arg(args.pfr_list, [0.2, 0.4, 0.6, 0.8])]

    return configs


def error_estimation():
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name).to('cuda')
        for model in args.models:
            for feature in available_mechanisms & set(args.features):
                experiment_name = f'error_{dataset_name}_{model}_{feature}'
                with PandasLogger(save_dir='results', experiment_name=experiment_name) as logger:
                    for eps in epsilons if args.epsilons is None else args.epsilons:
                        for run in range(args.repeats):
                            params = {
                                'task': 'error',
                                'dataset': dataset_name,
                                'model': model,
                                'feature': feature,
                                'rfr': 0,
                                'pfr': 1,
                                'eps': eps,
                                'run': run
                            }

                            params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
                            print(Fore.BLUE + params_str + Style.RESET_ALL)
                            logger.log_params(params)

                            data = privatize(dataset, method=feature, rfr=0, pfr=1, eps=eps)
                            t = ErrorEstimation(data=data, orig_features=dataset.x)
                            t.run(logger)


def prediction(task):
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, split_edges=(task == 'link'))
        dataset = dataset.to('cuda')

        for model in args.models:
            for feature in args.features:
                experiment_name = f'{task}_{dataset_name}_{model}_{feature}'
                with PandasLogger(save_dir='results', experiment_name=experiment_name) as logger:
                    for config in generate_config(feature):
                        for run in range(args.repeats):
                            params = {
                                'task': task,
                                'dataset': dataset_name,
                                'model': model,
                                'feature': feature,
                                **config,
                                'run': run
                            }

                            params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
                            print(Fore.BLUE + params_str + Style.RESET_ALL)
                            logger.log_params(params)

                            data = dataset
                            if feature in available_mechanisms:
                                data = privatize(data, method=feature, **config)

                            t = LearningTask(task_name=task, data=data, model_name=model)
                            t.run(logger)


def main():
    for task in args.tasks:
        if task in LearningTask.task_list():
            prediction(task)
        elif task == Task.ErrorEstimation:
            error_estimation()


if __name__ == '__main__':
    seed_everything(12345)

    # default values
    save_dir = './results'
    raw_feature_ratio = 0.2
    private_feature_ratios = [0, .2, .4, .6, .8, 1]
    epsilons = [1, 3, 5, 7, 9]
    epsilons_limited = [3, 5, 7]
    task_choices = Task.task_list()
    dataset_choices = get_available_datasets()
    model_choices = ['gcn']
    feature_choices = ['raw'] + list(available_mechanisms)

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-t', '--tasks', nargs='*', choices=task_choices, default=task_choices)
    parser.add_argument('-d', '--datasets', nargs='*', choices=dataset_choices, default=dataset_choices)
    parser.add_argument('-m', '--models', nargs='*', choices=model_choices, default=model_choices)
    parser.add_argument('-f', '--features', nargs='*', choices=feature_choices, default=feature_choices)
    parser.add_argument('-r', '--repeats', type=int, default=10)
    parser.add_argument('-o', '--output', type=str, default=save_dir)
    parser.add_argument('--pfr', nargs='*', type=float, dest='pfr_list')
    parser.add_argument('--rfr', nargs='*', type=float, dest='rfr_list')
    parser.add_argument('--eps', nargs='*', type=float, dest='eps_list')
    args = parser.parse_args()
    print(args)

    start = time.time()
    main()
    end = time.time()
    print('Done in', end - start, 'seconds.')
