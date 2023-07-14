
import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

pwd = os.getcwd()


class Results:
    def __init__(self,  path):
        self.path = path
        self.output_graph_path = os.path.join(path, "output_graph.png")
        self.csv_path = os.path.join(path, "output.csv")
        self.data = self.load_csv_to_numpy()
        self.names = self.data[:, 0]
        self.times = self.data[:, 1]
        self.errors = self.data[:, 2]
        self.streaks = self.data[:, 3]
        self.integral = self.data[:, 4]
        self.integral_filtered = self.data[:, 5]
        if len(self.data[0]) > 6:
            self.data_counts = self.data[:, 6]
        else:
            print("regrade all with image counts")
            exit(0)

    def load_csv_to_numpy(self):
        with open(self.csv_path, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            data = np.array(data)
        #sort data by column 0
        data = data[data[:,0].argsort()]
        return data

    def plot_graph(self):
        for i, data in enumerate(self.data_counts):
            plt.scatter(self.data_counts[i], self.integral_filtered[i], label=self.names[i])
        plt.savefig(self.output_graph_path)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    results = Results( os.path.join(pwd, "experiments"))
    results.plot_graph()