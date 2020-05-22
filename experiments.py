import time
from loggers import CustomMLFlowLogger
from argparse import ArgumentParser
from colorama import Fore, Style
from datasets import load_dataset, get_available_datasets
from mechanisms import privatize, available_mechanisms
from tasks import LearningTask, ErrorEstimation, Task
from pytorch_lightning import seed_everything


def get_ratios(feature, rfr, pfr_list):
    result = {(0, 1)}
    if feature == 'bit':
        result |= {(1-pfr, pfr) for pfr in pfr_list} | {(rfr, pfr) for pfr in pfr_list}
    return sorted(list(result))


def get_eps_list(feature, pfr):
    if feature == 'raw':
        return [None]
    if feature == 'bit' and pfr < 1:
        return set(epsilons_priv_ratio) & set(args.epsilons)
    else:
        return args.epsilons


def error_estimation():
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name).to('cuda')
        for model in args.models:
            for feature in available_mechanisms & set(args.features):
                for pfr in set(args.pfr_list) | {1}:
                    for eps in get_eps_list(feature, pfr):
                        for run in range(args.repeats):
                            print(
                                Fore.BLUE +
                                f'\ntask=error / dataset={dataset_name} / model={model} / '
                                f'feature={feature} / pfr={pfr} / eps={eps} / run={run}'
                                + Style.RESET_ALL
                            )

                            data = privatize(dataset, method=feature, rfr=1-pfr, pfr=pfr, eps=eps)
                            t = ErrorEstimation(data=data, orig_features=dataset.x)
                            result = t.run()
                            # results.append((f'gcn+{feature}', pnr, pfr, 1, eps, run, result))

            # save_results('error', dataset_name, 'gcn', feature, results, args.output)


def prediction(task):
    # init experiment logger
    logger = CustomMLFlowLogger(experiment_name='dpgcn-prediction')

    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, split_edges=(task == 'link'))
        dataset = dataset.to('cuda')

        for model in args.models:
            for feature in args.features:
                for rfr, pfr in get_ratios(feature, args.raw_feature_ratio, args.pfr_list):
                    for eps in get_eps_list(feature, pfr):
                        params = {
                            'task': task,
                            'dataset': dataset_name,
                            'model': model,
                            'feature': feature,
                            'rfr': rfr,
                            'pfr': pfr,
                            'eps': eps,
                        }

                        params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
                        logger.delete_runs(filter_string=f"tags.params='{params_str}'")

                        for run in range(args.repeats):
                            params['run'] = run

                            print(Fore.BLUE + params_str + f' | run={run}' + Style.RESET_ALL)

                            logger.create_run(tags={'params': params_str})
                            logger.log_params(params)

                            data = dataset
                            if feature in available_mechanisms:
                                data = privatize(data, method=feature, pfr=pfr, rfr=rfr, eps=eps)

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
    raw_feature_ratio = 0.2
    private_feature_ratios = [.2, .4, .6, .8]
    epsilons = [1, 3, 5, 7, 9]
    epsilons_priv_ratio = [3, 5, 7]
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
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument('--pfr', nargs='*', type=float, default=private_feature_ratios, dest='pfr_list')
    parser.add_argument('--rfr', type=float, default=raw_feature_ratio, dest='raw_feature_ratio')
    parser.add_argument('--eps', nargs='*', type=float, default=epsilons, dest='epsilons')
    args = parser.parse_args()
    print(args)

    start = time.time()
    main()
    end = time.time()
    print('Done in', end - start, 'seconds.')
