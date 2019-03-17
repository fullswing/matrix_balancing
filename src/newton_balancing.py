# Quasi Newton
import numpy as np 
import copy, time
import numpy as np
import copy, time, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import queue
import argparse
from numpy import genfromtxt
from collections import deque

def hessian(x, A):
    row, col = A.shape
    row_scale = x[0:row]
    col_scale = x[row:]
    diag_row_scale = np.diag(np.exp(row_scale))
    diag_col_scale = np.diag(np.exp(col_scale))
    H1 = np.diag(diag_row_scale.dot(A).dot(np.exp(col_scale)))
    H4 = np.diag(diag_col_scale.dot(A.T).dot(np.exp(row_scale)))
    H2 = np.diag(np.exp(row_scale)).dot(A).dot(np.diag(np.exp(col_scale)))
    H3 = H2.T
    tmp1 = np.concatenate((H1, H3),axis=0)
    tmp2 = np.concatenate((H2, H4), axis=0)
    H = np.concatenate((tmp1, tmp2), axis=1)
    return H
