default_early_stop_params = {
    'nodeclass': {
        'monitor': 'val_loss',
        'patience': 20,
        'min_delta': 0.0,
    },
    'linkpred': {
        'monitor': 'val_loss',
        'patience': 10,
        'min_delta': 0.0,
    }
}

default_trainer_params = {
    'nodeclass': {
        'max_epochs': 500,
        'min_epochs': 10
    },
    'linkpred': {
        'max_epochs': 500,
        'min_epochs': 100,
        'check_val_every_n_epoch': 10
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
            'lr': 0.001,
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
    },
    'twitch': {
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
    'proteins': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001,
            'dropout': 0.15
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0
        }
    },
    'bitcoin': {
        'nodeclass': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0,
            'dropout': 0
        },
        'linkpred': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.0001
        }
    }
}


def get_params(section, task, dataset, model_name):
    if section == 'trainer':
        return default_trainer_params[task]
    elif section == 'model':
        if model_name == 'gcn':
            return gcn_params[dataset][task]
    elif section == 'early-stop':
        if model_name == 'gcn':
            return default_early_stop_params[task]
