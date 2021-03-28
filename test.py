import pandas as pd
import torch
from datasets import load_dataset, supported_datasets
from mechanisms import OptimizedUnaryEncoding, RandomizedResopnse
from torch_sparse import matmul, SparseTensor
from tqdm.auto import tqdm
import seaborn as sns
sns.set_theme()


def randomized_response(adj, y, num_classes, eps, k):
    num_nodes = adj.size(0)
    b = RandomizedResopnse(eps=eps, d=num_classes)(y)
    deg = adj.sum(dim=1)
    nodes = torch.arange(num_nodes, device=deg.device)
    D_inv = SparseTensor(row=nodes, col=nodes, value=1 / deg)

    for i in range(k):
        b = matmul(adj, b, reduce='sum')
        b = matmul(D_inv, b)

    return (y == b.argmax(dim=1)).float().mean().item()


def unary_encoding(data, eps, k):
    b = OptimizedUnaryEncoding(eps=eps, d=data.num_classes)(data.y)
    deg = data.adj_t.sum(dim=1)
    nodes = torch.arange(data.num_nodes, device=deg.device)
    D_inv = SparseTensor(row=nodes, col=nodes, value=1 / deg)

    for i in range(k):
        b = matmul(data.adj_t, b, reduce='sum')
        b = matmul(D_inv, b)

    y = b.argmax(dim=1)
    return (y == data.y).float().mean().item()


def main():
    repeats = 100
    eps_range = [0.1, 0.5, 1, 2, 3]
    k_range = range(0, 17, 2)
    df = pd.DataFrame()
    for dataset in tqdm(supported_datasets):
        data = load_dataset(dataset).to('cuda')
        for method in tqdm([randomized_response, unary_encoding], leave=False, position=1):
            for eps in tqdm(eps_range, leave=False, position=2):
                for k in tqdm(k_range, leave=False, position=3):
                    rows = [
                        (
                            dataset,
                            method.__name__,
                            eps,
                            k,
                            method(data, eps, k=k)
                        ) for _ in tqdm(range(repeats), leave=False, position=4)
                    ]
                    df = df.append(
                        pd.DataFrame(rows, columns=['data', 'method', 'e', 'k', 'accuracy']), ignore_index=True
                    )

    df.to_pickle('test.pkl')


if __name__ == '__main__':
    main()
