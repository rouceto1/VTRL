# copy command: arash@chrono_server:/mnt/hdd/fn/grief_jpg/stromovkaold/*.csv ./ToBePlotted/stromovkaold
from inspect import getcallargs
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from time import time, ctime
import sys
import itertools
from scipy.interpolate import interp1d
import argparse
import pdb

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--in_path', type=str, help="path to the folder which csvs are located. BECAREFULL everything in this path will be plotted so dont put anything else than what you want to plot")
# parser.add_argument('--out_path', type=str, help="path to the location where you want your results to be saved")
# parser.add_argument('--label', type=str, help="if set to logic, expected legends will be shown. if set to file_name, file name will be shown as legend")
# args = parser.parse_args()
# out_path=args.out_path
# in_path=args.in_path
# label=args.label

parser = argparse.ArgumentParser(description='example: python error_analyser.py --dataset carlevaris --out_path ./Plotted')
parser.add_argument('--dataset', type=str, help="name of dataset: nordland stromovka carlevaris nordland_test: which is comparison of gts")
parser.add_argument('--out_path', type=str, help="path to the location where you want your results to be saved")
args = parser.parse_args()
dataset=args.dataset
out_path=args.out_path

# base_path='/home/arash/Desktop/workdir/IROS/stromovka_plot_data_server'
base_path='/mnt/hdd/fn/grief_jpg/stromovka'
stromovka={ "FM+NN first iteration" : base_path+"/0-1-fast-grief-1-hist.csv" ,
            "FM+NN second iteration" : base_path+"/1-2-fast-grief-1-hist.csv",
            "FM+NN third iteration" : base_path+"/2-3-fast-grief-1-hist.csv",
            "NN first iteration" : base_path+"/neural-0-1-fast-grief-1-hist.csv",
            "NN second iteration" : base_path+"/neural-1-2-fast-grief-1-hist.csv",
            "NN third iteration" : base_path+"/neural-2-3-fast-grief-1-hist.csv",
            "grief matcher" : base_path+"/pure-fast-grief-1-hist.csv",
}

nordland={ "FM+NN first iteration" : "/mnt/hdd/fn/nordland_rectified/0-1/fast-grief-hist.csv" ,
            "FM+NN second iteration" : "/mnt/hdd/fn/nordland_rectified/1-2/fast-grief-hist.cs",
            "FM+NN third iteration" : "/mnt/hdd/fn/nordland_rectified/2-3/fast-grief-hist.csv",
            "NN third iteration" : "/mnt/hdd/fn/nordland_rectified/2-3/nn_fast_grief_ORIGINAL_SORT.csv",
            "grief matcher" : "",
}


nordland_test={ "0-1-fast-grief-gt-hist" : "/mnt/hdd/fn/nordland_rectified/ground_truth_plots/0-1-fast-grief-gt-hist.csv" ,
            "0-1-fast-grief-hist.csv" : "/mnt/hdd/fn/nordland_rectified/ground_truth_plots/0-1-fast-grief-hist.csv",
            "1-2-fast-grief-gt-hist.csv" : "/mnt/hdd/fn/nordland_rectified/ground_truth_plots/1-2-fast-grief-gt-hist.csv",
            "1-2-fast-grief-hist.csv" : "/mnt/hdd/fn/nordland_rectified/ground_truth_plots/1-2-fast-grief-hist.csv",
            "2-3-fast-grief-gt-hist.csv" : "/mnt/hdd/fn/nordland_rectified/ground_truth_plots/2-3-fast-grief-gt-hist.csv",
            "2-3-fast-grief-hist.csv" : "/mnt/hdd/fn/nordland_rectified/ground_truth_plots/2-3-fast-grief-hist.csv",
}

carlevaris={ "FM+NN first iteration" : "/mnt/hdd/fn/grief_jpg/carlevaris/0-1-fast-grief-1-hist.csv" ,
            "FM+NN second iteration" : "/mnt/hdd/fn/grief_jpg/carlevaris/1-2-fast-grief-1-hist.csv",
            "FM+NN third iteration" : "/mnt/hdd/fn/grief_jpg/carlevaris/2-3-fast-grief-1-hist.csv",
            "NN third iteration" : "/mnt/hdd/fn/grief_jpg/carlevaris/neural-2-3-fast-grief-1-hist.csv",
            "grief matcher" : "/mnt/hdd/fn/grief_jpg/carlevaris/pure-fast-grief-1-hist.csv",
}
def date_str_gen():
    return ctime(time()).replace(':','_').replace(" ","_")

def get_cdf(array,bins=10000):
    array=np.abs(array)
    count, bins_count = np.histogram(array,bins=bins)
    pdf = count / count.sum()
    cdf = np.cumsum(pdf)
    return [cdf,bins_count[1:]]

if __name__=="__main__":  
    data=[]
    ax1=0
    ax2=0
    fig,[ax1,ax2]=plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(7,5))
    if dataset=='stromovka':
      DATASET=stromovka
      bin_count=500
      xlim=bin_count
    elif dataset=='carlevaris':
      DATASET=carlevaris
      bin_count=500
      xlim=300
    elif dataset=='nordland':
      DATASET=nordland
      bin_count=10000
      xlim=1000
    elif dataset=='nordland_test':
      DATASET=nordland_test
      bin_count=10000
      xlim=1000
    else:
      raise NameError("[-] error in dataset name passed")
    
    for key in DATASET:
        # pdb.set_trace()
        try:
          arr=np.loadtxt(DATASET[key],delimiter=',')[:,2]
        except Exception as E:
          print("[-] error in reading file {} exception {}".format(DATASET[key],E))
          exit(1)
        cdf,bin_count=get_cdf(arr,bin_count)
        if key=="FM+NN first iteration" or key=="FM+NN second iteration" or key=="FM+NN third iteration":
          ax1.plot(bin_count,cdf,label=key)
        if key=="FM+NN third iteration" or key=="NN third iteration" or key=="grief matcher" :
          ax2.plot(bin_count,cdf,label=key)
        elif dataset=="nordland_test":
          # plt.plot(cdf[0:xlim],label=key)
          ax1.plot(cdf,label=key)
    ax1.legend()
    ax2.legend()
    # plt.legend()
    title=dataset.capitalize()
    plt.suptitle(title)
    ax1.grid()
    ax2.grid()
    # plt.grid()
    plt.ylim(0.6,1)
    ax1.set_ylabel("Prob. of correct registration [-]")
    ax1.set_xlabel("Registration error threshold [px]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path,'latest'+'.png'))
    plt.savefig(os.path.join(out_path,date_str_gen()+'.png'))
    plt.show()
