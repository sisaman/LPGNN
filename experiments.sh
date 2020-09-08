# Error estimation
python error.py -d cora elliptic facebook github twitch -m gm mbm -e 0.5 1 2 4 -k 1 2 4 8 -a gcn mean

# Node Classification
### Cora
python train.py -d cora -m raw mbm gm -e 0.5 1 2 4 -k 1 2 4 8 --aggs gcn mean --lr 0.01 --weight-decay 0.01 -r 5

### Twitch
python train.py -d twitch -m raw mbm gm -e 0.5 1 2 4 -k 1 2 4 8 --aggs gcn mean --lr 0.001 --weight-decay 0.01 --dropout 0.5 -r 5

### Facebook
python train.py -d facebook -m raw mbm gm -e 0.5 1 2 4 -k 1 2 4 8 --aggs gcn mean --lr 0.01 --weight-decay 0.001 --dropout 0.5 -r 5

### Github
python train.py -d github -m raw mbm gm -e 0.5 1 2 4 -k 1 2 4 8 --aggs gcn mean --lr 0.01 --dropout 0.5 -r 5

### Elliptic
python train.py -d elliptic -m raw mbm gm -e 0.5 1 2 4 -k 1 2 4 8 --aggs gcn mean --lr 0.01 --dropout 0.5 -r 5





