import pandas as pd
import torch
from torch.nn.functional import nll_loss
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import accuracy
from tqdm import tqdm, trange

from datasets import load_dataset
from gnn import GCN
from utils import one_bit_response

torch.manual_seed(12345)

setup = {
    'datasets': [
        'cora',
        # 'citeseer',
        # 'pubmed',
        # 'reddit',
        # 'ppi',
        # 'flickr',
        # 'yelp',
    ],
    'methods': [
        'private',
        'default',
        'localdegree',
        'random',
    ],
    'eps': [
        0.1,
        # 0.2,
        # 0.5,
        1,
        3,
        5,
        7,
        9,
    ],
    'hidden_dim': 16,
    'epochs': 200,
    'repeats': 10,
}


def node_classification():
    device = torch.device('cuda')
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = []
        dataset = load_dataset(dataset_name)
        for run in trange(setup['repeats'], desc='Run', leave=False):
            for method in tqdm(setup['methods'], desc='Method', leave=False):
                for epsilon in tqdm(setup['eps'] if method.startswith('private') else [0], desc='Epsilon',
                                    leave=False):

                    data = dataset[0]

                    with torch.no_grad():
                        if method == 'one-hot':
                            data.x = torch.eye(data.num_nodes)
                        elif method == 'random':
                            data.x = torch.rand(data.num_nodes, data.num_node_features)*data.delta + data.alpha
                        elif method == 'localdegree':
                            data.x = None
                            data.num_nodes = len(data.y)
                            data = LocalDegreeProfile()(data)
                        elif method.startswith('private'):
                            data = one_bit_response(data, epsilon)

                        data = data.to(device)

                    model = GCN(
                        input_dim=data.num_node_features,
                        output_dim=dataset.num_classes,
                        hidden_dim=setup['hidden_dim'],
                        private=(method.startswith('private')),
                        epsilon=epsilon,
                        alpha=data.alpha,
                        delta=data.delta,
                    ).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

                    model.train()
                    for epoch in trange(
                            setup['epochs'],
                            desc=f'Epoch (dataset={dataset_name}, run={run}, method={method}, eps={epsilon})',
                            leave=False):

                        optimizer.zero_grad()
                        out = model(data)
                        loss = nll_loss(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        model.eval()
                        pred = model(data).argmax(dim=1)
                        acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

                        if not method.startswith('private'):
                            for eps in setup['eps']:
                                results.append((run, method, eps, acc))
                        else:
                            results.append((run, method, epsilon, acc))

        df_result = pd.DataFrame(data=results, columns=['run', 'conf', 'eps', 'acc'])
        df_result.to_pickle(f'results/node_classification_{dataset_name}.pkl')


if __name__ == '__main__':
    node_classification()
