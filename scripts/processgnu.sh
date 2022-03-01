dir=/mnt/hdd/fn/nordland_rectified
dir2=/mnt/hdd/fn/grief_jpg/stromovkaold
dir3=/mnt/hdd/fn/grief_jpg/carlevaris

cat $dir/0-1/fast-grief-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/nPFM01.srt
cat $dir/0-2/fast-grief-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/nFMNN02w1.srt
cat $dir/0-3/fast-grief-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/nFMNN03w2.srt
cat $dir/0-3/nn-fast-grief.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/nPNNw2.srt

cat $dir2/pure-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/sPFM.srt
cat $dir2/0-1-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/sFMNNw1.srt
cat $dir2/1-2-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/sFMNNw2.srt
cat $dir2/2-3-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/sFMNNw3.srt

cat $dir3/pure-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/cPFM.srt
cat $dir3/0-1-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/cFMNNw1.srt
cat $dir3/1-2-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/cFMNNw2.srt
cat $dir3/2-3-fast-grief-1-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../results/cFMNNw3.srt

