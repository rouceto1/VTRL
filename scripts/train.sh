#!/bin/bash

#Sh script for training of NN on nordland datastet.

######## paramteras
imwidth=512
dataset_path=/mnt/hdd/fn/nordland_rectified/
descriptor=fast
detector=grief
suf= ##suffix if you want to train different type 
#########


batch_name1=0
batch_name2=1

name=$descriptor-$detector
dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-hist$suf.csv

nmodel=$dataset_path/$batch_name1-$batch_name2-$name$suf.pt
date
./../FM/tools/match_all $descriptor $detector $dataset_folder - - 0 1 14124 $imwidth $displacement
python ../NN/NNT.py --in_model_path start --data_path $dataset_folder --d0 season_00 --d1 season_01 --dsp $displacement --omodel $nmodel
date
batch_name1=0
batch_name2=2

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-hist$suf.csv
model=$nmodel
nmodel=$dataset_path$batch_name1-$batch_name2-$name$suf.pt
python ../NN/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_folder/nn-$name$suf.csv

./../FM/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name$suf.csv 1 1 14124 $imwidth $displacement

python ../NN/NNT.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --dsp $displacement --omodel $nmodel
date

batch_name1=1
batch_name2=2

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-hist$suf.csv
python ../NN/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_folder/nn-$name$suf.csv
./../FM/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name$suf.csv 1 1 14124 $imwidth $displacement
model=$nmodel
nmodel=$dataset_path$batch_name1-$batch_name2-$name$suf.pt
python ../NN/NNT.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --dsp $displacement --omodel $nmodel


batch_name1=0
batch_name2=3

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-hist$suf.csv
model=$nmodel
nmodel=$dataset_path$batch_name1-$batch_name2-$name$suf.pt
python ../NN/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_folder/nn-$name$suf.csv
./../FM/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name$suf.csv 1 1 14124 $imwidth $displacement
python ../NN/NNT.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --dsp $displacement --omodel $nmodel

batch_name1=1
batch_name2=3

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-hist$suf.csv
python ../NN/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_folder/nn-$name$suf.csv
./../FM/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name$suf.csv 1 1 14124 $imwidth $displacement
tmodel=$nmodel
nmodel=$dataset_path$batch_name1-$batch_name2-$name$suf.pt
python ../NN/NNT.py --in_model_path $tmodel --data_path $dataset_folder --d0 season_00 --d1 season_01 --dsp $displacement --omodel $nmodel

batch_name1=2
batch_name2=3

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-hist$suf.csv
python ../NN/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_folder/nn-$name$suf.csv
./../FM/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name$suf.csv 1 1 14124 $imwidth $displacement
tmodel=$nmodel
nmodel=$dataset_path$batch_name1-$batch_name2-$name$suf.pt
python ../NN/NNT.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --dsp $displacement --omodel $nmodel
