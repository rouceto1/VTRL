imwidth=512


dataset_path=/mnt/hdd/fn/nordland_rectified/


descriptor=fast
detector=grief

batch_name1=0
batch_name2=3

name=$descriptor-$detector


dataset_folder=$dataset_path/0-1/
displacement=$dataset_path/0-1-$name-test.csv
model=$dataset_path/0-1-$name.pt

#python alignment/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_path/0-1/0-1-nn-$name.csv
echo FM on first model spring fall
#./grief-experimental/tools/match_all $descriptor $detector $dataset_folder - 0-1-nn-$name.csv 1 1 14124 $imwidth $displacement

dataset_folder=$dataset_path/0-1/
displacement=$dataset_path/1-2-$name-test.csv
model=$dataset_path/1-2-$name.pt

#python alignment/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_path/0-1/1-2-nn-$name.csv
echo FM on first model spring fall
#./grief-experimental/tools/match_all $descriptor $detector $dataset_folder - 1-2-nn-$name.csv 1 1 14124 $imwidth $displacement

dataset_folder=$dataset_path/0-3/
displacement=$dataset_path/2-3-$name-test.csv
model=$dataset_path/2-3-$name.pt

python alignment/NNE.py --in_model_path $model --data_path $dataset_folder --d0 season_00 --d1 season_01 --ocsv $dataset_path/0-3/2-3-nn-$name.csv
echo FM on first model spring fall
./grief-experimental/tools/match_all $descriptor $detector $dataset_folder - 2-3-nn-$name.csv 1 1 14124 $imwidth $displacement
