#!/bin/bash

####### 
#dataset_path=/mnt/hdd/fn/grief_jpg/carlevaris
dataset_path=/mnt/hdd/fn/grief_jpg/stromovka
model_path=/mnt/hdd/fn/nordland_rectified/
dataset_path=/home/rouceto1/git/VTRL
model_path=/home/rouceto1/git/VTRL
hist_type=1
suf=
descriptor=fast
detector=grief
#########


name=$descriptor-$detector

nnoutput=$name-nn$suf.csv
fmoutput=$name-$hist_type-hist$suf.csv


iteration=1-2

displacement=$dataset_path/$iteration-$fmoutput
nndisplacement=$dataset_path/neural-$iteration-$fmoutput
displacement_pure=$dataset_path/pure-$fmoutput

./../build/FM_node $descriptor $detector $dataset_path - /$iteration-$nnoutput 0 $hist_type 539 1024 $displacement_pure ###this is for producing the pure FM output
python ../src/NN/NNE.py --in_model_path $model_path$iteration-$name$suf.pt --data_path $dataset_path --d0 season_00 --d1 season_01 --ocsv $dataset_path/$iteration-$nnoutput
./src/FM/tools/match_all $descriptor $detector $dataset_path - /$iteration-$nnoutput 1 $hist_type 539 1024 $displacement $nndisplacement

iteration=2-3
displacement=$dataset_path/$iteration-$fmoutput
nndisplacement=$dataset_path/neural-$iteration-$fmoutput

python ../src/NN/NNE.py --in_model_path $model_path$iteration-$name$suf.pt --data_path $dataset_path --d0 season_00 --d1 season_01 --ocsv $dataset_path/$iteration-$nnoutput
./../build/FM/tools/match_all $descriptor $detector $dataset_path - /$iteration-$nnoutput 1 $hist_type 539 1024 $displacement $nndisplacement

iteration=3-0
displacement=$dataset_path/$iteration-$fmoutput
nndisplacement=$dataset_path/neural-$iteration-$fmoutput
python src/NN/NNE.py --in_model_path $model_path$iteration-$name$suf.pt --data_path $dataset_path --d0 season_00 --d1 season_01 --ocsv $dataset_path/$iteration-$nnoutput
./src/FM/tools/match_all $descriptor $detector $dataset_path - /$iteration-$nnoutput 1 $hist_type 539 1024 $displacement $nndisplacement
 
