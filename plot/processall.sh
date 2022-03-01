dir=/mnt/hdd/fn/nordland_rectified
dir2=/mnt/hdd/fn/grief_jpg/stromovka
dir3=/mnt/hdd/fn/grief_jpg/carlevaris

type=-chrono
folder=resultsch

cat $dir/0-2/fast-grief-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/nFMNN02w1.srt
cat $dir/0-3/fast-grief-hist.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/nFMNN03w2.srt
cat $dir/0-3/pure-fast-grief-nn.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/nPNN03w2.srt

cat $dir/0-1/pure-fast-grief-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/nPFM01.srt
cat $dir/0-2/pure-fast-grief-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/nPFM02.srt
cat $dir/0-3/pure-fast-grief-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/nPFM03.srt

cat $dir2/pure-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/sPFM.srt
cat $dir2/0-1-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/sFMNNw1.srt
cat $dir2/1-2-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/sFMNNw2.srt
cat $dir2/2-3-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/sFMNNw3.srt

cat $dir3/pure-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/cPFM.srt
cat $dir3/0-1-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/cFMNNw1.srt
cat $dir3/1-2-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/cFMNNw3.srt
cat $dir3/2-3-fast-grief-1-hist$type.csv |cut -f 3 -d ,|tr -d -|sort -n > ../$folder/cFMNNw2.srt

