REP=10
DIR="./temp"

cd ..
python train.py -d "$DB" -m mbm -e 1 -k 1 2 4 8 16 32 --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR
python train.py -d "$DB" -m mbm -e 0.1 0.5 1 2 4 -k 1 2 4 8 16 32 --no-loops --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR
python train.py -d "$DB" -l 0.2 0.4 0.6 0.8 1.0 -m mbm -e 1 -k 2 4 8 --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR --no-loops
python train.py -d "$DB" -l 0.2 0.4 0.6 0.8 1.0 -m mbm -e 0.5 2.0 -k 8 --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR --no-loops
python train.py -d "$DB" -m raw -k 1 --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR
python train.py -d "$DB" -m raw -k 1 2 4 8 16 32 --no-loops --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR
python train.py -d "$DB" -m rnd -k 1 --lr "$LR" --wd "$WD" --dp "$DP" -r $REP -o $DIR
