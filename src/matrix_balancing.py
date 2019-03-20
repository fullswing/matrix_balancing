import numpy as np
import copy, time
import numpy as np
import copy, time, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import argparse

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

def sinkhorn(A):
    A = np.array(A)
    row, col = A.shape
    scale_row = [1/sum(A[r]) for r in range(row)]
    A = (A.T * scale_row).T
    scale_col = [1/sum(A[:,c]) for c in range(col)]
    A = A * scale_col
    return A

def objective_function(x, A):
    n, m = A.shape
    row_scale = np.array(x[0:n])
    col_scale = np.array(x[n:])
    return np.exp(row_scale).dot(A).dot(np.exp(col_scale)) - sum(row_scale) - sum(col_scale)

def barrier_gradient(x, A):
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

def capacity_gradient(x, A):
    const_value = A.dot(np.exp(x))
    n, m = A.shape
    grad = np.array([sum(A[:,j]/const_value)*np.exp(x[j])-1 for j in range(m)])
    return grad

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

def lbfgs_gradient_descent(x, A, gradient=barrier_gradient, m=10, e=1e-6, max_iter=80, truncation=True):
    k = 0
    s = deque()
    y = deque()
    grad = gradient(x, A)
    l = []
    lr = 0.1
    min_lr = 0.0001
    while True:
        if truncation and np.linalg.norm(grad) < e:
            break
        if not truncation and k > max_iter:
            break
        if k > max_iter:
            print("Gradient Descent did not converge.")
            return x
        d = two_loop_recursion(grad, s, y)
        x = np.array(x) + lr * np.array(d)
        lr = max(lr*0.8, min_lr)
        loss = np.linalg.norm(grad)
        
        l.append(loss)
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
    print("total steps:", k)
    return x, l

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
    H = H + 1e-5 * np.eye(len(H))
    return H

def newton_gradient_descent(x, A, m=10, e=1e-6, max_iter=80, truncation=True):
    k = 0
    s = deque()
    y = deque()
    grad = barrier_gradient(x, A)
    l = []
    lr = 1.0
    while True:
        if truncation and np.linalg.norm(grad) < e:
            break
        if not truncation and k > max_iter:
            break
        H = hessian(x, A)
        d = -np.linalg.inv(H).dot(grad)
        x = np.array(x) + lr * np.array(d)
        loss = np.linalg.norm(grad)
        l.append(loss)
        if k % 5 == 0:
            print("step:{} norm:{}".format(k, np.linalg.norm(grad)))
        newGrad = barrier_gradient(x, A)
        if len(s) == m:
            s.popleft()
            y.popleft()
        s.append(d)
        y.append(newGrad-grad)
        grad = newGrad
        k += 1
    print("total steps:", k)
    return x, l

def matrix_balancing(A, algorithm='lbfgs', objective='barrier', truncation=True):
    row, col = A.shape
    if algorithm == 'lbfgs':
        if objective == 'barrier':
            x = -np.ones(row+col)
            start = time.time()
            x, l = lbfgs_gradient_descent(x, A, gradient=barrier_gradient,truncation=truncation)
            row_scale = x[0:row]
            col_scale = x[row:]
            result_mat = np.diag(np.exp(row_scale)).dot(A).dot(np.diag(np.exp(col_scale)))
            end = time.time()
            print("Elapsed time for lbfgs-barrier balancing:{} sec.".format(end-start))
            return result_mat, l
        elif objective == 'capacity':
            x = -np.ones(row)
            start = time.time()
            x, l = lbfgs_gradient_descent(x, A, gradient=capacity_gradient,truncation=truncation)
            const_value = A.dot(np.exp(x))
            result_mat = (A.dot(np.diag(np.exp(x))).T/const_value).T
            end = time.time()
            print("Elapsed time for lbfgs-capacity balancing:{} sec.".format(end-start))
            return result_mat, l
    elif algorithm == 'newton':
        x = -np.ones(row+col)
        start = time.time()
        x, l = newton_gradient_descent(x, A, truncation=truncation)
        row_scale = x[0:row]
        col_scale = x[row:]
        result_mat = np.diag(np.exp(row_scale)).dot(A).dot(np.diag(np.exp(col_scale)))
        end = time.time()
        print("Elapsed time for newton-barrier balancing:{} sec.".format(end-start))
        return result_mat, l