# Error estimation
python error.py -d cora elliptic facebook github mit twitch -m gm mbm -e 0.25 0.5 1 2 4 8 -k 1 2 5 10 20 50 100 200 -a gcn mean

# Node Classification
### Cora
python train.py -d cora -m raw --lr 0.01 --weight-decay 0.01 -r 10

### MIT
python train.py -d mit -m raw --lr 0.01 --dropout 0.5  -r 10

### Twitch
python train.py -d twitch -m raw --lr 0.001 --weight-decay 0.01 --dropout 0.5 -r 10

### Facebook
python train.py -d facebook -m raw --lr 0.01 --weight-decay 0.001 --dropout 0.5 -r 10

### Github
python train.py -d github -m raw --lr 0.01 --dropout 0.5 -r 10

### Elliptic
python train.py -d elliptic -m raw --lr 0.01 --dropout 0.5 -r 10





