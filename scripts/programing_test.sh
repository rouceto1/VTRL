#!/bin/bash

####### 
#dataset_path=/mnt/hdd/fn/grief_jpg/carlevaris
dataset_path=/mnt/hdd/fn/grief_jpg/stromovka
model_path=/mnt/hdd/fn/nordland_rectified/

dataset_path=/home/rouceto1/git/VTRL/grief_jpg/stromovka
model_path=/home/rouceto1/git/VTRL/nordland_rectified/

hist_type=1
suf=
descriptor=fast
detector=grief
#########


name=$descriptor-$detector

nnoutput=$name-nn$suf.csv
fmoutput=$name-$hist_type-hist$suf.csv


iteration=1-2

displacement=$dataset_path/$iteration-$fmoutput-new

displacement_2=$dataset_path/$iteration-$fmoutput-old

displacement_3=$dataset_path/$iteration-$fmoutput-new2
nndisplacement=$dataset_path/neural-$iteration-$fmoutput
displacement_pure=$dataset_path/pure-$fmoutput-new
displacement_pure_2=$dataset_path/pure-$fmoutput-old

displacement_pure_3=$dataset_path/pure-$fmoutput-new2
#./../src/FM/src/match_all $descriptor $detector $dataset_path - /$iteration-$nnoutput 0 $hist_type 539 1024 $displacement_pure_2

#./../build/FM_new $descriptor $detector $dataset_path - /$iteration-$nnoutput 0 $hist_type 539 1024 $displacement_pure_3 ###this is for producing the pure FM output
#./../build/FM_node $descriptor $detector $dataset_path - /$iteration-$nnoutput 0 $hist_type 539 1024 $displacement_pure ###this is for producing the pure FM output
 python3 ../src/NN/NNE.py --in_model_path $model_path$iteration-$name$suf.pt --data_path $dataset_path --d0 season_00 --d1 season_01 --ocsv $dataset_path/$iteration-$nnoutput

#./../build/FM_node $descriptor $detector $dataset_path - /$iteration-$nnoutput 1 $hist_type 539 1024 $displacement $nndisplacement ###this is for producing the pure FM output
 ./../build/FM_new $descriptor $detector $dataset_path - /$iteration-$nnoutput 1 $hist_type 539 1024 $displacement_3 $nndisplacement ###this is for producing the pure FM output
#./../src/FM/src/match_all $descriptor $detector $dataset_path - /$iteration-$nnoutput 1 $hist_type 539 1024 $displacement_2 $nndisplacement

#diff $displacement_pure_2 $displacement_pure 

diff $displacement_pure_3 $displacement_pure -s
#diff $displacement_2 $displacement 

diff $displacement_3 $displacement -s

