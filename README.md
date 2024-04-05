# CURRENTLY under construction until RAL paper decision

# Visual Teach Repeat and Learn (VTRL) pipeline

This pipeline was made as an investigation into self-learning capabilities for mobile robotics.

The pipeline uses Feature Matching (FM) to establish pixel shifts between images and then feeds this info to the siamese Neural Network (NN) for training.
For each subsequent drive through the environment, the NN provides course pixel shift probabilities which are used to enhance the FM.
This scheme continues by incrementally teaching the NN from the acquired data.
IT also continues to explore the enviroment.

## Datasets

To run this pipeline download the datasets below.

All have to be in one folder and this has to be in 'datasets" folder in the home directory of this porject.
Can be done by linking it ` ln -s dataset_folder_path `

- [Cestlice](https://datasets.chronorobotics.tk/s/wdq7P8K6tx16aRA)
- [Withm Warf](https://datasets.chronorobotics.tk/s/afc5YEubVEtzBhd)

- [Ground Truth](https://datasets.chronorobotics.tk/s/wPp7NRA1boukvdk)


## Prerequisites

The code was prepared to run on Ubuntu 20 with conda. Install conda and other prerequisities.
To make conda enviroment use

```
conda env create --file environment.yaml
```

While it is possible to train the NN without powerful GPU, the training should be done on a card with enough VRAM (8GB).
Without training the pipeline runs on any hardware but benefits the most from multithreaded CPUs.

In any scripts mentioned change the path at their begging to your location of the datasets.

To build the whole c++ part use Cmake in a folder build.

```
mkdir build
cd build
cmake ..
make -j 255
```

## Running

Several scripts are provieded.

### Training

The make_and_process_strategy.py creates desired missions based on specifications in the file.
It creates a folder for each mission in the "experiments" directory and saves the 0 iteration.training

After missions are generated ./process.py can be run to fully process (teach, and evaluate) all given missions.
It is than expected to use "data_backup.sh" to backup all the computed data. 

### Grading

In compare_results.py user has to point to home folder whhere all the data are stored.
Results loading reauqires to name a specific folder in the file loading.

It outputs plots as well as CSV file with data for statistics.
These files are than loaded using "get_stats.py" which returns all the statistics.
