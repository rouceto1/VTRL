imwidth=512
dataset_path=/mnt/hdd/fn/nordland_rectified/
descriptor=fast
detector=grief

batch_name1=0
batch_name2=1

name=$descriptor-$detector

#first data
dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-nn-a.csv
nndisplacement=$dataset_folder/pure-$name-nn.csv
nmodel=$dataset_path/$batch_name1-$batch_name2-$name.pt
#./grief-experimental/tools/match_all $descriptor $detector $dataset_folder - nn-$name.csv 1 1 14124 $imwidth $displacement $nndisplacement

batch_name1=0
batch_name2=2

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/$name-nn-a.csv

nndisplacement=$dataset_folder/pure-$name-nn.csv
nmodel=$dataset_path/$batch_name1-$batch_name2-$name.pt
#./grief-experimental/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name.csv 1 1 14124 $imwidth $displacement $nndisplacement

#winter 
batch_name1=0
batch_name2=3

nmodel=$dataset_path/$batch_name1-$batch_name2-$name.pt
dataset_folder=$dataset_path$batch_name1-$batch_name2

nndisplacement=$dataset_folder/pure-$name-nn.csv
displacement=$dataset_folder/$name-nn-a.csv
./grief-experimental/tools/match_all $descriptor $detector $dataset_folder/ - nn-$name.csv 1 1 14124 $imwidth $displacement $nndisplacement

