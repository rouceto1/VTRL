import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

pwd = os.getcwd()
import colorsys


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
        self.t = self.get_N_HexCol(len(self.data))

        self.data = self.data[np.array(self.data[:, self.data_counts_filtered], dtype=int).argsort()]

    def load_csv_to_numpy(self):
        with open(self.csv_path, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            data = np.array(data)
        # sort data by column 0
        data = data[data[:, 0].argsort()]
        return data

    def get_N_HexCol(self, N=5):
        HSV_tuples = [(x * 1.0 / (N * 1.5), 1, 1) for x in range(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append('#%02x%02x%02x' % tuple(rgb))
        return hex_out

    def plot_scatter(self, filter=False):
        fig = plt.figure()
        ax = plt.gca()
        for i, data in enumerate(self.data):
            if float(self.data[i][self.data_counts_filtered]) > 2000:
                pass
                if filter:
                    continue
            ax.scatter(float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral_filtered]),
                       label=self.data[i][self.names] + " " + self.data[i][self.data_counts_filtered], c=self.t[i],
                       alpha=0.5)
            ax.annotate(self.data[i][self.names],
                        (float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral_filtered])))
        plt.title(self.csv_name)

        # plt.ylim([0.43, 0.455]
        # ax.set_yscale('log')
        plt.xlabel("Integral")
        plt.ylabel("Data count")
        # plt.legend()
        plt.savefig(self.output_graph_path)
        plt.show()

    def plot_lines(self):
        # loads and plots AC_lines from line.pkl for each folder
        fig = plt.figure()
        ax = plt.gca()
        for i, data in enumerate(self.data):
            if float(self.data[i][self.data_counts_filtered]) > 2000:
                pass
                # continue
            with open(os.path.join(self.path, self.data[i][self.names], "line.pkl"), 'rb') as f:
                line = pickle.load(f)
            ax.plot(line[0], line[1], label=self.data[i][self.data_counts_filtered], c=self.t[i], alpha=0.5)
        plt.legend()
        plt.title("AC lines for different counts of teaching data images")
        plt.xlabel("allowed image shift")
        plt.ylabel("probability of correct registration")
        plt.show()


if __name__ == "__main__":
    results = Results(os.path.join(pwd, "backups", "strands-strands-6-2"), "output.csv")
    results.plot_scatter()
    results.plot_scatter(filter=True)
    results.plot_lines()
