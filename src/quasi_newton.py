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
from scipy.optimize import line_search
from sk_balancing import sinkhorn

def preprocess(A):
    A = np.array(A)
    print(A)
    n, m = A.shape
    if n == m:
        A += 1e-5 * np.eye(n)
    delete_row_index = []
    for i in range(n):
        if len(A[i]) > 0 and np.count_nonzero(A[i]) <= 1:
            delete_row_index.append(i)
    A = np.delete(A, delete_row_index, 0)
    print("deleted row:", np.array(delete_row_index))
    delete_col_index = []
    _, m = A.shape
    for i in range(m):
        if len(A[:,i]) > 0 and np.count_nonzero(A[:,i]) <= 1:
            delete_col_index.append(i)
    A = np.delete(A, delete_col_index, 1)
    print("deleted col:", np.array(delete_col_index))
    print(A.shape)
    #A = sinkhorn(A)
    return A

def objective_function(x, A):
    n, m = A.shape
    row_scale = np.array(x[0:n])
    col_scale = np.array(x[n:])
    #print(row_scale, col_scale)
    return np.exp(row_scale).dot(A).dot(np.exp(col_scale)) - sum(row_scale) - sum(col_scale)

def gradient(x, A):
    row, col = A.shape
    #g = []
    g = np.zeros(row+col)
    scale_row = np.array(x[0:row])
    assert len(scale_row) == row
    scale_col = np.array(x[row:])
    assert len(scale_col) == col
    hist = []
    for i in range(row):
        var = 0
        for j in range(col):
            var += A[i][j] * np.exp(scale_col[j])
            if var == float('inf'):
                print(A[i][j], np.exp(scale_col[j]), scale_col[j], j)
            assert var != float('inf')
            #hist.append(np.exp(scale_col[j]))
        var *= np.exp(scale_row[i])
        var -= 1
        g[i] = var
        #g.append(var)
    for j in range(col):
        var = 0
        for i in range(row):
            var += A[i][j] * np.exp(scale_row[i])
            if var == float('inf'):
                print(A[i][j], np.exp(scale_row[i]), scale_row[i], i)
            assert var != float('inf')
            #hist.append(np.exp(scale_row[i]))
        var *= np.exp(scale_col[j])
        var -= 1
        #g.append(var)
        g[row+j] = var
    #print(x[0:row] - x[row:])
    return np.array(g)

def two_loop_recursion(grad, s, y):
    n = len(s)
    d = -grad
    if n == 0:
        return 0.01 * d
    a = []
    hist = []
    for i in reversed(range(n)):
        alpha = s[i].dot(d) / s[i].dot(y[i])
        v = s[i].dot(y[i])
        hist.append(v)
        d = np.array(d) - alpha * y[i]
        a.append(alpha)

    d = (s[n-1].dot(y[n-1]) / y[n-1].dot(y[n-1])) * d
    a = a[::-1]
    
    for i in range(n):
        b = y[i].dot(d) / s[i].dot(y[i])
        d = d + s[i] * (a[i] - b)
    return d

def l_bfgs(x, A, m=10, e=1e-6, max_iter=100, prefix='hic', truncation=True):
    k = 0
    s = deque()
    y = deque()
    #vectorized_gradient = np.vectorize(gradient)
    grad = gradient(x, A)
    #print(A)
    #print(x)
    #grad = vectorized_gradient(x, A)
    row, col = A.shape
    assert len(grad) == len(A) + len(A[0])
    l = []
    norm_hist = []
    lr = 0.1
    init = 0.1
    min_lr = 0.0001
    min_loss = 1e+10
    #f1 = open('../result/'+prefix+'_loss_with_annealing.txt', 'w')
    #f2 = open('../result/'+prefix+'lr_history_for_hic.txt', 'w')
    while True:
        if truncation and np.linalg.norm(grad) < e:
            break
        elif not truncation and k > max_iter:
            break
        d = two_loop_recursion(grad, s, y)
        tmp = np.linalg.norm(grad)
        #f1.write(str(tmp) + '\n')
        #print(x)
        #gfk = gradient(x, A)
        #alpha, fc, gc, new_fval, old_fval, new_slope = line_search(f=objective_function,myfprime=gradient,gfk=gfk,args=(A,),xk=x,pk=d,amax=1)
        #alpha = None
        #if alpha == None:
        #    x = np.array(x) + min(lr*0.8, min_lr) * np.array(d)
        #else:
        #    print("alpha:", alpha)
        #    x = np.array(x) + alpha * np.array(d)
        x = np.array(x) + lr * np.array(d)
        lr = max(lr*0.8, min_lr)
        #f2.write(str(lr) + '\n')
        
        loss = np.linalg.norm(grad)
        """
        if loss > 3 * min_loss:
            lr = init
        else:
            min_loss = loss
        """
        l.append(loss)
        norm_hist.append(np.linalg.norm(grad))
        if k % 5 == 0:
            print("step:{} norm:{}".format(k, np.linalg.norm(grad)))

        newGrad = gradient(x, A)
        #newGrad = vectorized_gradient(x, A)
        if len(s) == m:
            s.popleft()
            y.popleft()
        s.append(d)
        #print("y:", newGrad-grad)
        y.append(newGrad-grad)
        grad = newGrad
        k += 1
    #print(np.diag(np.exp(x)).dot(A).dot(np.exp(x)))
    #print("loss:", loss)
    #f1.close()
    #f2.close()
    #print(np.exp(x))
    row_scale = x[0:row]
    col_scale = x[row:]    
    print("total steps:", k)
    result_mat = np.diag(np.exp(row_scale)).dot(A).dot(np.diag(np.exp(col_scale)))
    assert result_mat.shape == A.shape
    #print(k)
    return x, l, result_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", help = "Target matrix data")
    parser.add_argument("filetype", help="File type: hic or csv", type=str)
    parser.add_argument("output", help="Graph between loss and step num", type=str)
    parser.add_argument("balanced", help="Balanced matrix data", default=True, type=str)
    parser.add_argument("--skiprows", help="Skip rows", default=0, type=int)
    parser.add_argument("--delimiter", help="Delimiter", default="", type=str)
    parser.add_argument("--max_iter", help="Max number of iteration for L-BFGS algorithm", default=80, type=int)
    parser.add_argument("--truncation", help="Truncation is activated with this option", action='store_true')
    parser.add_argument("--preprocess", help="Preprocess the target matrix with this option", action='store_true')
    args = parser.parse_args()
    if args.filetype == "hic":
        mat = np.loadtxt(args.matrix, skiprows=args.skiprows)
    elif args.filetype == "csv":
        mat = np.genfromtxt(args.matrix, delimiter=args.delimiter, skip_header=args.skiprows)
    else:
        print("Error:Invalid file type", file=sys.stderr)
    trg = mat
    if args.preprocess:
        print("Preprocessing...")
        trg = preprocess(trg)
        print("Done!")
    n, m = trg.shape
    x = -np.ones(n+m)
    start = time.time()
    x, l, result = l_bfgs(x, trg, max_iter=args.max_iter,truncation=args.truncation)
    end = time.time()
    print("elapsed time:{} sec.".format(end-start))
    plt.plot(l)
    plt.yscale('log')
    plt.ylabel('loss')
    plt.xlabel('steps')
    plt.xticks(list(range(0,len(l), 10)))
    plt.legend
    plt.savefig(args.output)
    np.savetxt(args.balanced, result, delimiter=",")
    plt.clf()

if __name__ == '__main__':
    main()
