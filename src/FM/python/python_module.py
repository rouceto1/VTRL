import os

import numpy as np
import ctypes as ct
from pathlib import Path

libPath = Path("build/libaligment.so").absolute()
print(libPath)

lib = ct.cdll.LoadLibrary(libPath)

_doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

lib.teachOnFiles.restype = None
lib.teachOnFiles.argtypes = [
    ct.POINTER(ct.c_char_p),  # cosnt char **
    ct.POINTER(ct.c_char_p),  # cosnt char **
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # numpy float array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ct.c_int
]
lib.evalOnFiles.restype = None
lib.evalOnFiles.argtypes = [
    ct.POINTER(ct.c_char_p),  # cosnt char **
    ct.POINTER(ct.c_char_p),  # cosnt char **
    _doublepp,
    _doublepp,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # numpy float array
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ct.c_int
]


def cpp_teach_on_files(combinations, displacement, feature_count_l, feature_count_r, matches, length):
    """
    objective: to get displacement and feature counts on 'length' of given image pairs
    """
    strArrayType = ct.c_char_p * length
    c1 = strArrayType()
    c2 = strArrayType()
    for i, param in enumerate(combinations):
        if (i == length):
            break
        # print(param[0])
        # print (c1[i])
        c1[i] = param[0].encode('utf-8')
        c2[i] = param[1].encode('utf-8')

    lib.teachOnFiles(c1, c2, displacement, feature_count_l, feature_count_r, matches, length)
    return displacement, feature_count_l, feature_count_r


##CAN RETURN LARGE DISPLACEMENTS IF NO MATCHES ARE FOUND. These should be in thousnends of percents larger then possible
def cpp_eval_on_files(combinations, displacement, feature_count_l, feature_count_r, matches, length, hist_in, hist_out, GT):
    """
    objective: to get displacement and feature counts on 'length' of given image pairs
    includes using given histogram(or probabilty distribution) to prefer specific alignments
    return:
    displacement between the images in pixels,
    average total feature count on both images,
    histogram containing all matches
    """
    strArrayType = ct.c_char_p * length
    c1 = strArrayType()
    c2 = strArrayType()
    for i, param in enumerate(combinations):
        if (i == length):
            break
        c1[i] = param[0].encode('utf-8')
        c2[i] = param[1].encode('utf-8')
    hi = (hist_in.__array_interface__['data'][0]
          + np.arange(hist_in.shape[0]) * hist_in.strides[0]).astype(np.uintp)
    ho = (hist_out.__array_interface__['data'][0]
          + np.arange(hist_out.shape[0]) * hist_out.strides[0]).astype(np.uintp)
    # length = ct.c_int(hist_in.shape[0])
    width = ct.c_int(hist_in.shape[1])
    lib.evalOnFiles(c1, c2, hi, ho, GT, displacement, feature_count_l, feature_count_r, matches, width, length)
    return displacement, feature_count_l, feature_count_r, hist_out
