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

        self.data = self.data[np.array(self.data[:, self.data_counts_filtered], dtype=int).argsort()]

    def load_csv_to_numpy(self):
        with open(self.csv_path, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            data = np.array(data)
        # sort data by column 0
        data = data[data[:, 0].argsort()]
        return data

    def get_N_HexCol(self,N=5):
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append('#%02x%02x%02x' % tuple(rgb))
        return hex_out


    def plot_graph(self):
        t = self.get_N_HexCol(len(self.data))
        fig = plt.figure()
        ax = plt.gca()
        for i, data in enumerate(self.data):
            if float(self.data[i][self.data_counts_filtered]) > 8200:
                pass
                #continue
            ax.scatter( float(self.data[i][self.integral_filtered]), float(self.data[i][self.data_counts_filtered]),
                        label=self.data[i][self.names] + " "+  self.data[i][self.data_counts_filtered], c=t[i])
            ax.annotate(self.data[i][self.names], (float(self.data[i][self.integral_filtered]), float(self.data[i][self.data_counts_filtered])))
        plt.title(self.csv_name)

        #plt.ylim([0.43, 0.455]
        ax.set_yscale('log')
        plt.legend()
        plt.savefig(self.output_graph_path)

        plt.show()


if __name__ == "__main__":
    results = Results(os.path.join(pwd, "experiments"), "output.csv")
    results.plot_graph()
