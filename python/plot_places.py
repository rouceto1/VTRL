#!/usr/bin/env python3
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt


dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"

def plot_places(dist):

    chosen_positions_file = "input.txt"
    chosen_positions = np.loadtxt(os.path.join(dist, chosen_positions_file),int)
    #print (chosen_positions)
    interpolated = []
    for pose in chosen_positions:
        for i in range(60):
            interpolated.append(pose)
    locs, labels = plt.xticks()
    #labels = locs/60
    print(locs)
    print(labels)
    #plt.xticks(locs, labels.astype(int))
    plt.plot(interpolated)
    plt.ylabel("Place visited", fontsize = 18)
    plt.xlabel("Timestamp [s]", fontsize = 18)

    plt.savefig(os.path.join(dist,"places.png"))


if __name__ == "__main__":
    plot_places(dataset_path)
