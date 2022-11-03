import src.NN.NNET as neuralka
import numpy as np
import os
import pickle
import sys
from helper_functions import *
import src.FM.python.python_module as grief
def NN_eval(file_list_nn,weights_file):
    a, b, hist_in, dsp = neuralka.NNeval_from_python(np.array(file_list_nn),"strands", weights_file)
    hist_in = np.float64(hist_in)
    return hist_in, dsp

def FM_NN_eval(file_list,filetype_NN,filetype_FM,weights_file,dataset_path,cache2,use_cache,gt):
    count = len(file_list)
    count = 10
    disp = np.zeros(count, dtype = np.float32)
    fcount = np.zeros(count, dtype = np.int32)
    gt = np.zeros(count, dtype = np.float64)
    file_list = np.array(file_list)[:count]

    if not os.path.exists(cache2) or not use_cache:
        hist_in, displacement = NN_eval(choose_proper_filetype(filetype_NN, file_list),os.path.join(dataset_path, weights_file))
        with open(cache2, 'wb') as handle:
            pickle.dump(hist_in, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("making chache" + str(cache2))
    else:
        print("reading cache" + str(cache2))
        with open(cache2, 'rb') as handle:
            hist_in = pickle.load(handle)


    hist_out = np.zeros((count,63), dtype = np.float64)
    displacements,feature_count,histograms = grief.cpp_eval_on_files(choose_proper_filetype(filetype_FM,file_list),
                                        disp, fcount, count, hist_in, hist_out, gt)
    FM_out = np.array([disp, fcount], dtype=np.float64).T
    file_list = np.array(file_list)[:count]
    np.set_printoptions(threshold=sys.maxsize)
    return displacements,feature_count,histograms



## Feature matcher treis to evaluate on all files in the list 
def FM_eval (file_list,filetype_FM):
    count = len(file_list)
    count = 10 ## THIS IS TO LIMIT IT FOR DEBUGING PURPOUSES (may be a fnction in the future?)
    disp = np.zeros(count, dtype = np.float32)
    fcount = np.zeros(count, dtype = np.int32)

    grief.cpp_teach_on_files(choose_proper_filetype(filetype_FM,file_list),
                             disp,fcount,count)
    FM_out = np.array([disp, fcount], dtype=np.float32).T
    #file_list.append(disp)
    file_list = np.array(file_list)[:count]
    files_with_displacement = np.append(file_list, FM_out, axis=1)
    return files_with_displacement


