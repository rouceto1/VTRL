# Visual Teach Repeat and Learn (VTRL) pipeline

This pipeline was made as an investigation into self-learning capabilities for mobile robotics.
A paper about this was submitted to Sensors Journal.

The pipeline uses Feature Matching (FM) to establish pixel shifts between images and then feeds this info to the siamese Neural Network (NN) for training.
For each subsequent drive through the environment, the NN provides course pixel shift probabilities which are used to enhance the FM.
This scheme continues by incrementally teaching the NN from the acquired data.

## Datasets

To run this pipeline download the datasets below.

[Testing](https://datasets.chronorobotics.tk/s/QUeUFeUen0942t9)
[Training - NORDLAND](https://datasets.chronorobotics.tk/s/aVD7YOTvtOirYhU)
Unzip them and note the path.

If one wishes not to train but only to use a NN without training, the Training set contains weights that were used for experiments in the paper.

## Prerequisites

The code was prepared to run on Ubuntu 20 with conda. Install conda and other prerequisities.
To make conda enviroment use

```
conda env create --file environment.yaml
```


While it is possible to train the NN without powerful GPU, the training should be done on a card with enough VRAM (6GB).
Without training the pipeline runs on any hardware but benefits the most from multithreaded CPUs.

In any scripts mentioned change the path at their begging to your location of the datasets.

To build the whole c++ part use Cmake in a folder build.

```
mkdir build
cd build
cmake ..
make -j 255
```
--- everything past this is ODL--- 

## Running
 Several scripts are proviede. 
 Running test.py will annotatea a dataset using known weights, teach new weights and evalutes these waeights.


### training

For training run 'train.sh' script. This script uses the Nordland dataset to get trained.
The trained weights are written to the root folder of the dataset used with '.pt' suffix this name also contains descriptor-detector and custom suffix for distinction.
The training iterates over the numbered folders and computes the weights.
Possible results of computations are saved in these folders as well such as the output of FM or NN into CSV files which allows resuming wherever the pipeline ended last time.

### Testing

For testing run 'test.sh' script.
This script tests the learned weights on Stromovka or Carlevaris datasets depending on the configuration available at the beginning of the file.
The output of the FM and NN for each iteration is then dumped into the folder of each dataset.

## Evaluation

Each FM outputs a CSV file that contains some information in a format of
|Estimated pixel shift|Match features|Error of shift to GT| error rate at 35 pixels | all bins of the internal histogram |
where each row corresponds to an image pair from the dataset.
For each FM+NN combo as well as solo NN if any was used as an input.

Each NNE outputs a CSV file that contains for each image pair a line that has probability distribution of the shift.

The names of these files contain info: which iteration of weights were used, detector, descriptor
if it is present then weights were trined Purley on GT,
if neural is the prefix of the file then this is the output of the NN only.

If the name contains the word pure then the CSV is from pure FM only.

NOTES:

If you wanna use C_types for python.... you need to wrap all that you wanna expose in the H file into
extern "C"{
void foo.....
}

create python_module where you defince input and output from the fuction. nupy arrays are ok, standart stuf si ok vectors are pain.
in this file you also define function that you call from python. stadart python input and tha you there process whateer are your python datatypes into C datatypes. Only arrays are somewhat needed and strings. all strings have to have the encoding.
it is recomanded to pass pointers to stuff you alocate in python for return functins from these if you want multiple returns, touples are pain.

to get get stuff wroking link dataset folder to the main VTRL folder using
ln -s "dataset_folder_path"
