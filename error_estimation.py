import pandas as pd
import torch
from tqdm import tqdm, trange

from datasets import load_dataset
from gnn import GCNConv, GConvDP
from utils import get_degree, one_bit_response

setup = {
    'datasets': [
        # 'cora',
        # 'citeseer',
        'pubmed',
        'flickr',
        # 'reddit',
        # 'ppi',
        # 'yelp',
    ],
    # 'methods': [
    #     'private',
    # ],
    'eps': [
        0.1,
        0.2,
        0.5,
        1,
        2,
        # 3,
        5,
        # 7,
        # 9,
    ],
    'hidden_dim': 16,
    'repeats': 50,
}


@torch.no_grad()
def error_estimation():
    device = torch.device('cuda')
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = None
        dataset = load_dataset(dataset_name)
        data = dataset[0].to(device)

        delta = data.delta.clone()
        delta[delta == 0] = 1  # avoid inf and nan
        delta = delta

        gcnconv = GCNConv().to(device)
        gc = gcnconv(data.x, data.edge_index)
        gconvdp = GConvDP(epsilon=0, alpha=data.alpha, delta=data.delta).to(device)

        for run in trange(setup['repeats'], desc='Run', leave=False):
            for epsilon in tqdm(setup['eps'], desc=f'Epsilon (dataset={dataset_name}, run={run})', leave=False):

                data = one_bit_response(dataset[0], epsilon).to(device)
                gconvdp.eps = epsilon
                gc_hat = gconvdp(data.x, data.edge_index)

                diff = (gc - gc_hat) / delta
                error = torch.norm(diff, p=1, dim=1) / diff.shape[1]
                deg = get_degree(data)

                res = torch.cat([
                    torch.tensor([epsilon] * deg.shape[0]).view(-1, 1).float().to(device),
                    error.view(-1, 1),
                    deg.view(-1, 1),
                ], dim=1)

                if results is None:
                    results = res
                else:
                    results = torch.cat([results, res])

        df_result = pd.DataFrame(data=results.cpu().numpy(), columns=['eps', 'err', 'deg'])
        df_result.to_pickle(f'results/error_estimation_{dataset_name}.pkl')


if __name__ == '__main__':
    error_estimation()
