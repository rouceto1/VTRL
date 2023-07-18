import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

pwd = os.getcwd()


class Results:
    def __init__(self, path, csv_name):
        self.path = path
        self.csv_name = csv_name
        self.output_graph_path = os.path.join(path, self.csv_name + ".png")
        self.csv_path = os.path.join(path, self.csv_name)
        self.data = self.load_csv_to_numpy()
        self.names = 0
        self.times = 1
        self.errors = 2
        self.streaks = 3
        self.integral = 4
        self.integral_filtered = 5
        self.data_counts = 6
        self.data_counts_filtered = 7

        self.data = self.data[self.data[:, self.data_counts_filtered].argsort()]

    def load_csv_to_numpy(self):
        with open(self.csv_path, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            data = np.array(data)
        # sort data by column 0
        data = data[data[:, 0].argsort()]
        return data

    def plot_graph(self):
        t = ['#646B63', '#8E402A', '#343B29', '#AF2B1E', '#9C9C9C', '#00BB2D', '#F39F18', '#9B111E', '#8673A1',
             '#293133']
        for i, data in enumerate(self.data):
            if float(self.data[i][self.data_counts_filtered]) > 8200:
                pass
                #continue
            plt.scatter(float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral_filtered]),
                        label=self.data[i][self.names], c=t[i])
        plt.title(self.csv_name)

        plt.ylim([0.43,0.455])
        plt.legend()
        plt.savefig(self.output_graph_path)

        plt.show()


if __name__ == "__main__":
    results = Results(os.path.join(pwd, "experiments"), "output.csv")
    results.plot_graph()
    results = Results(os.path.join(pwd, "experiments"), "10k.csv")
    results.plot_graph()
