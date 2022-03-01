imwidth=512
dataset_path=/mnt/hdd/fn/nordland_rectified/
descriptor=fast
detector=grief

pre=-med

batch_name1=0
batch_name2=1

name=$descriptor-$detector

#first data
dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/pure-$name-hist$pre.csv

./grief-experimental/tools/match_all $descriptor $detector $dataset_folder - - 0 1 14124 $imwidth $displacement

batch_name1=0
batch_name2=2

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/pure-$name-hist$pre.csv

./grief-experimental/tools/match_all $descriptor $detector $dataset_folder/ - - 0 1 14124 $imwidth $displacement

#winter 
batch_name1=0
batch_name2=3

dataset_folder=$dataset_path$batch_name1-$batch_name2
displacement=$dataset_folder/pure-$name-hist$pre.csv
./grief-experimental/tools/match_all $descriptor $detector $dataset_folder/ - - 0 1 14124 $imwidth $displacement

