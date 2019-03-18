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
from newton_balancing import hessian

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
    return np.exp(row_scale).dot(A).dot(np.exp(col_scale)) - sum(row_scale) - sum(col_scale)

def gradient(x, A):
    row, col = A.shape
    g1 = np.zeros(row)
    g2 = np.zeros(col)
    scale_row = np.array(x[0:row])
    assert len(scale_row) == row
    scale_col = np.array(x[row:])
    assert len(scale_col) == col
    g1 = A.dot(np.exp(scale_col)) * np.exp(scale_row) - 1
    g2 = A.T.dot(np.exp(scale_row)) * np.exp(scale_col) - 1
    g = np.append(g1,g2)
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

<<<<<<< HEAD
def gradient_descent(x, A, m=10, e=1e-6, max_iter=100, prefix='hic', truncation=True):
    # Optimizing by L-BFGS algorithm
=======
def gradient_descent(x, A, m=10, e=1e-6, max_iter=100, prefix='hic', truncation=True, algorithm="lbfgs"):
>>>>>>> newton
    k = 0
    s = deque()
    y = deque()
    grad = gradient(x, A)
    row, _ = A.shape
    assert len(grad) == len(A) + len(A[0])
    l = []
    norm_hist = []
    if algorithm == "lbfgs":
        lr = 0.1
    elif algorithm == "newton":
        lr = 1.0
    min_lr = 0.0001
    while True:
        if truncation and np.linalg.norm(grad) < e:
            break
        elif not truncation and k > max_iter:
            break
        if algorithm == "lbfgs":
            d = two_loop_recursion(grad, s, y)
        elif algorithm == "newton":
            H = hessian(x, A)
            d = -np.linalg.inv(H).dot(grad)
        x = np.array(x) + lr * np.array(d)
        if algorithm == "newton":
            #lr, _, _, _, _, _ = line_search(objective_function, gradient, x, pk=d, args=(A,), amax=50)
            lr = 1.0
            if lr != 1.0:
                print(lr)
        elif algorithm == "lbfgs":
            lr = max(lr*0.8, min_lr)
        loss = np.linalg.norm(grad)
        l.append(loss)
        norm_hist.append(np.linalg.norm(grad))
        if k % 5 == 0:
            print("step:{} norm:{}".format(k, np.linalg.norm(grad)))

        newGrad = gradient(x, A)
        if len(s) == m:
            s.popleft()
            y.popleft()
        s.append(d)
        y.append(newGrad-grad)
        grad = newGrad
        k += 1
    row_scale = x[0:row]
    col_scale = x[row:]
    print("total steps:", k)
    result_mat = np.diag(np.exp(row_scale)).dot(A).dot(np.diag(np.exp(col_scale)))
    assert result_mat.shape == A.shape
    return x, l, result_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", help = "Target matrix data")
    parser.add_argument("filetype", help="File type: hic or csv", type=str)
    parser.add_argument("output", help="Graph between loss and step num", type=str)
    parser.add_argument("balanced", help="Balanced matrix data", default=True, type=str)
    parser.add_argument("--algorithm", help="Algorithm for gradient descent:lbfgs or newton", default="lbfgs", type=str)
    parser.add_argument("--skiprows", help="Skip rows", default=0, type=int)
    parser.add_argument("--delimiter", help="Delimiter", default="", type=str)
    parser.add_argument("--max_iter", help="Max number of iteration for L-BFGS algorithm", default=80, type=int)
    parser.add_argument("--truncation", help="Truncation is activated with this option", action='store_true')
    parser.add_argument("--preprocess", help="Preprocess the target matrix with this option", action='store_true')
    parser.add_argument("--sinkhorn", help="Run sinkhorn once and then apply optimization", action='store_true')
    args = parser.parse_args()
    if args.algorithm != "lbfgs" and args.algorithm != "newton":
        print("Error:Invalid optimization method", file=sys.stderr)
        return
    if args.filetype == "hic":
        mat = np.loadtxt(args.matrix, skiprows=args.skiprows)
    elif args.filetype == "csv":
        mat = np.genfromtxt(args.matrix, delimiter=args.delimiter, skip_header=args.skiprows)
    else:
        print("Error:Invalid file type", file=sys.stderr)
        return
    trg = mat
    if args.preprocess:
        print("Preprocessing...")
        trg = preprocess(trg)
        print("Done!")
    if args.sinkhorn:
        print("Running Sinkhorn once")
        trg = sinkhorn(trg)
        print("Done!")
    n, m = trg.shape
    x = -np.ones(n+m)
    start = time.time()
<<<<<<< HEAD
    x, l, result = gradient_descent(x, trg, max_iter=args.max_iter,truncation=args.truncation)
=======
    x, l, result = gradient_descent(x, trg, max_iter=args.max_iter,truncation=args.truncation, algorithm=args.algorithm)
>>>>>>> newton
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