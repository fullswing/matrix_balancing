# Sinkhorn knopp
import numpy as np 
import copy, time
import numpy as np
import copy, time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import queue
from collections import deque
 
def sinkhorn(A):
    A = np.array(A)
    row, col = A.shape
    for r in range(row):
        var = sum(A[r])
        #print(var)
        for c in range(col):
            A[r][c] = A[r][c] / var
    for c in range(col):
        var = sum(A[:,c])
        for r in range(row):
            A[r][c] = A[r][c] / var
    return A

if __name__ == '__main__':
    m = np.loadtxt('../data/HIC_%s_1000000_%s.txt.gz' % ('k562', 'exp'), skiprows=1)
    th = 1e-6
    loss = 1
    step = 0
    l = []

    trg = copy.copy(m)
    #trg = np.array([[1,3,2],[3,4,5],[2,5,4]], np.float32)
    #trg = np.array([[1,2,3],[2,0,0.9],[3,0.9,5]], float)
    row, col = trg.shape
    s = time.time()
    while loss > th:
        loss = 0
        for r in range(row):
            var = sum(trg[r])
            #print(var)
            for c in range(col):
                trg[r][c] = trg[r][c] / var
        for c in range(col):
            var = sum(trg[:,c])
            for r in range(row):
                trg[r][c] = trg[r][c] / var
        for r in range(row):
            loss += np.abs(sum(trg[r]) - 1) ** 2
        for c in range(col):
            loss += np.abs(sum(trg[:,c]) - 1) ** 2
        #print(loss)
        l.append(loss)
        #print(loss)
        # calculate loss
    #    loss = np.sum([np.abs(np.sum(1 - np.sum(r))) for r in trg])
        #tmp = trg.T
    #    loss += np.sum([np.abs(np.sum(1 - np.sum(c))) for c in tmp])
        step += 1
        if step % 5 == 0:
            #print(step)
            plt.imshow(1/(trg+1e-3),cmap='RdBu')
    #        plt.show()
            print(loss)
    #        print(trg)
    t = time.time()
    print(t-s)
    print(trg)
