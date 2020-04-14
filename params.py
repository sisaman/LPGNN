default_early_stop_params = {
    'monitor': 'val_loss',
    'patience': 20,
    'min_delta': 0.0,
}

default_trainer_params = {
    'nodeclass': {
        'max_epochs': 500,
        'min_epochs': 10
    },
    'linkpred': {
        'max_epochs': 500,
        'min_epochs': 100
    }
}

node2vec_params = {
    'model': {
        'embedding_dim': 128,
        'walk_length': 80,
        'context_size': 10,
        'walks_per_node': 10,
        'batch_size': 1,
        'lr': 0.01,
        'weight_decay': 0
    },
    'trainer': {
        'max_epochs': 1,
    }
}

gcn_params = {
    'cora': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01,
            'dropout': 0.5
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01
        }
    },
    'citeseer': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.1,
            'dropout': 0.5
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01
        }
    },
    'pubmed': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001,
            'dropout': 0.5
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01
        }
    },
    'flickr': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001
        }
    },
    'amazon-photo': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0
        }
    },
    'amazon-computers': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001
        }
    }
}


def get_params(section, task, dataset, model):
    if section == 'trainer':
        if model == 'node2vec':
            return node2vec_params['trainer']
        else:
            return default_trainer_params[task]
    elif section == 'model':
        if model == 'node2vec':
            return node2vec_params['model']
        else:
            return gcn_params[dataset][task]
    elif section == 'early-stop':
        if model == 'node2vec':
            return {}
        else:
            return default_early_stop_params
