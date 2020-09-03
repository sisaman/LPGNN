# Error estimation
python error.py -d citeseer cora elliptic flickr twitch -m lm pm pgc -e 1 3 5 7 9

# Node Classification
### CiteSeer
python train.py -t node -d citeseer -m raw -e 0 --learning-rate 0.01 --weight-decay 0.1 --dropout 0.5 -r 10
python train.py -t node -d citeseer -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.1 --dropout 0.5 -r 10
### Cora
python train.py -t node -d cora -m raw -e 0 --learning-rate 0.01 --weight-decay 0.01 --dropout 0.5 -r 10
python train.py -t node -d cora -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.01 --dropout 0.5 -r 10
### Twitch
python train.py -t node -d twitch -m raw -e 0 --learning-rate 0.001 --weight-decay 0.0001 --dropout 0 -r 10
python train.py -t node -d twitch -m lm pm pgc -e 1 5 9 --learning-rate 0.001 --weight-decay 0.0001 --dropout 0 -r 10
### Elliptic
python train.py -t node -d elliptic -m raw -e 0 --learning-rate 0.01 --weight-decay 0 --dropout 0 -r 10
python train.py -t node -d elliptic -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0 --dropout 0 -r 10
### Flickr
python train.py -t node -d flickr -m raw -e 0 --learning-rate 0.01 --weight-decay 0 --dropout 0 -r 10
python train.py -t node -d flickr -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0 --dropout 0 -r 10


# Link Prediction
### CiteSeer
python train.py -t link -d citeseer -m raw -e 0 --learning-rate 0.01 --weight-decay 0.01 -r 10
python train.py -t link -d citeseer -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.01 -r 10
### Cora
python train.py -t link -d cora -m raw -e 0 --learning-rate 0.01 --weight-decay 0.01 -r 10
python train.py -t link -d cora -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.01 -r 10
### Twitch
python train.py -t link -d twitch -m raw -e 0 --learning-rate 0.01 --weight-decay 0.001 -r 10
python train.py -t link -d twitch -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.001 -r 10
### Elliptic
python train.py -t link -d elliptic -m raw -e 0 --learning-rate 0.01 --weight-decay 0.0001 -r 10
python train.py -t link -d elliptic -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.0001 -r 10
### Flickr
python train.py -t link -d flickr -m raw -e 0 --learning-rate 0.01 --weight-decay 0.001 -r 10
python train.py -t link -d flickr -m lm pm pgc -e 1 5 9 --learning-rate 0.01 --weight-decay 0.001 -r 10
