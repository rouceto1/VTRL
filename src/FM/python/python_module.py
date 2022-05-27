import os

import numpy as np
import ctypes as ct
lib = ct.cdll.LoadLibrary("/datafast/VTRL/VTRL/build/libaligment.so")



lib.teachOnFiles.restype = None
lib.teachOnFiles.argtypes = [
    ct.POINTER(ct.c_char_p), # cosnt char **
    ct.POINTER(ct.c_char_p),# cosnt char **
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #numpy float array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ct.c_int
]
def cpp_teach_on_files(combinations,displacement, feature_count, length):
    """
    objective: to get displacement and feature counts on 'length' of given image pairs
    """
    strArrayType = ct.c_char_p * length
    c1 = strArrayType()
    c2 = strArrayType()
    for i, param in enumerate(combinations):
        if (i == length):
            break
        #print(param[0])
        #print (c1[i])
        c1[i] = param[0].encode('utf-8')
        c2[i] = param[1].encode('utf-8')

    print (c1)
    lib.teachOnFiles(c1,c2, displacement, feature_count, length)
    return displacement,feature_count
