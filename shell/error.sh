cd ..
python error.py -t eps -d cora citeseer pubmed facebook github lastfm -m agm obm mbm -e 0.5 1 2 4 -a mean gcn
python error.py -t deg -d cora citeseer pubmed facebook github lastfm -m agm obm mbm -e 1 -a mean gcn
