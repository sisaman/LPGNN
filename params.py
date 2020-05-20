default_early_stop_params = {
    'node': {
        'monitor': 'val_loss',
        'patience': 20,
        'min_delta': 0.0,
    },
    'link': {
        'monitor': 'val_loss',
        'patience': 10,
        'min_delta': 0.0,
    }
}

default_trainer_params = {
    'node': {
        'max_epochs': 500,
        'min_epochs': 10
    },
    'link': {
        'max_epochs': 500,
        'min_epochs': 100,
        'check_val_every_n_epoch': 10
    }
}

gcn_params = {
    'cora': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01,
            'dropout': 0.5
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01
        }
    },
    'citeseer': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.1,
            'dropout': 0.5
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01
        }
    },
    'pubmed': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001,
            'dropout': 0.5
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.01
        }
    },
    'flickr': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'link': {
            'output_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.001
        }
    },
    'amazon-photo': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0
        }
    },
    'amazon-computers': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001
        }
    },
    'twitch': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001
        }
    },
    'proteins': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0.001,
            'dropout': 0.15
        },
        'link': {
            'output_dim': 16,
            'lr': 0.01,
            'weight_decay': 0
        }
    },
    'bitcoin': {
        'node': {
            'hidden_dim': 16,
            'lr': 0.01,
            'weight_decay': 0,
            'dropout': 0
        },
        'link': {
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
