DB="facebook"
LR=0.01
WD=0.001
DP=0.5

python train.py -d $DB -m mbm -e 1 -k 1 2 4 8 16 32 --lr $LR --wd $WD --dp $DP -r 10
python train.py -d $DB -m mbm -e 0.1 0.5 1 2 4 -k 1 2 4 8 16 32 --no-loops --lr $LR --wd $WD --dp $DP -r 10
python train.py -d $DB -m raw -k 1 -d --lr $LR --wd $WD --dp $DP -r 10
python train.py -d $DB -m raw -k 1 2 4 8 16 32 --no-loops --lr $LR --wd $WD --dp $DP -r 10
python train.py -d $DB -m rnd -k 1 --lr $LR --wd $WD --dp $DP -r 10
