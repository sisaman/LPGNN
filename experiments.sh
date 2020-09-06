# Error estimation
python error.py -d citeseer cora elliptic flickr twitch -m lm pm pgc -e 1 3 5 7 9

# Node Classification
### Cora
python train.py -t node -d cora -m raw --lr 0.01 --weight-decay 0.01 --dropout 0 -r 10

### MIT
python train.py -t node -d mit -m raw --lr 0.01 --weight-decay 0.05 --dropout 0 -r 10

### Twitch
python train.py -t node -d twitch -m raw --lr 0.001 --weight-decay 0.01 --dropout 0.5 -r 10

### Facebook
python train.py -t node -d facebook -m raw --lr 0.01 --weight-decay 0.001 --dropout 0.5 -r 10

### Github
python train.py -t node -d github -m raw --lr 0.01 --weight-decay 0 --dropout 0.5 -r 10

### Elliptic
python train.py -t node -d elliptic -m raw --lr 0.01 --weight-decay 0 --dropout 0.5 -r 10





