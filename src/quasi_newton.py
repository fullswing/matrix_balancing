# Quasi Newton
import numpy as np 
import copy, time
import numpy as np
import copy, time
import matplotlib.pyplot as plt
import queue
from collections import deque

def preprocess(A):
    A = np.array(A)
    n, _ = A.shape
    #print(A)
    delete_row_index = []
    for i in range(n):
        if len(A[i]) > 0 and np.count_nonzero(A[i]) <= 1:
            delete_row_index.append(i)
    A = np.delete(A, delete_row_index, 0)
    #print(A)
    print("deleted row:", np.array(delete_row_index))
    delete_col_index = []
    _, m = A.shape
    for i in range(m):
        #print(np.count_nonzero(A[:,i]))
        if len(A[:,i]) > 0 and np.count_nonzero(A[:,i]) <= 1:
            delete_col_index.append(i)
    A = np.delete(A, delete_col_index, 1)
    #print(A)
    print("deleted col:", np.array(delete_col_index))
    print(A.shape)
    return A

def gradient(x, A):
    g = []
    n = len(x)
    hist = []
    for i in range(n):
        var = 0
        for j in range(n):
            #print(A[i][j] * np.exp(x[j]))
            var += A[i][j] * np.exp(x[j])
            #if i == j:
            #    var += A[i][j] * np.exp(x[j])
            if var == float('inf'):
                print(A[i][j], np.exp(x[j]), x[j], j)
            assert var != float('inf')
            hist.append(np.exp(x[j]))
        var *= np.exp(x[i])
        var -= 1
        g.append(var)
    #print("gradient:{}".format(np.array(g)))
    if g[0] == float('inf'):
        print(hist)
    return np.array(g)

def two_loop_recursion(grad, s, y):
    n = len(s)
    d = -grad
    if n == 0:
        return 0.01 * d
    a = []
    hist = []
    #for i in range(n):
    for i in reversed(range(n)):
        alpha = s[i].dot(d) / s[i].dot(y[i])
        v = s[i].dot(y[i])
        hist.append(v)
        #print("alpha:", alpha)
        d = np.array(d) - alpha * y[i]
        a.append(alpha)
        #print(len(s))
        #print(len(y))
    #d = s[n-1].dot(y[n-1]) / y[n-1].dot(y[n-1]) * d
    #print("corr:",s[-1].dot(y[-1]) / y[-1].dot(y[-1]))
    #print("bunbo:",y[-1].dot(y[-1])) 
    #print("bunshi:",s[-1].dot(y[-1]))
    d = (s[n-1].dot(y[n-1]) / y[n-1].dot(y[n-1])) * d
    #print(a)
    #print(hist)
    a = a[::-1]
    #print("hoge search vec:", d)
    #for i in reversed(range(n)):
    
    for i in range(n):
        #print(d)
        b = y[i].dot(d) / s[i].dot(y[i])
        #print(s[i])
        #print(a[i])
        d = d + s[i] * (a[i] - b)
        #print("search vec:", d, a[i], b)
    return d

def l_bfgs(x, A, m=10, e=1e-6, max_iter=100):
    k = 0
    s = deque()
    y = deque()
    grad = gradient(x, A)
    l = []
    norm_hist = []
    lr = 0.1
    init = 0.1
    min_lr = 0.0001
    min_loss = 1e+10
    f1 = open('../result/hic_loss_with_annealing.txt', 'w')
    f2 = open('../result/lr_history_for_hic.txt', 'w')
    while np.linalg.norm(grad) > e or k <= max_iter:
        d = two_loop_recursion(grad, s, y)
        #print("search vector:", d)
        tmp = np.linalg.norm(grad)
        f1.write(str(tmp) + '\n')
        x = np.array(x) + lr * np.array(d)
        lr = max(lr*0.8, min_lr)
        f2.write(str(lr) + '\n')
        
        #plt.hist(d)
        #plt.savefig("../result/" + str(k).zfill(2) + ".png")
        #plt.clf()
        loss = np.linalg.norm(np.diag(np.exp(x)).dot(A).dot(np.exp(x))-np.ones(len(x)))
        if loss > 3 * min_loss:
            lr = init
        else:
            min_loss = loss
        #if len(l) > 0 and loss < l[-1]:
            #print(l[-1])
            #print(loss, l[-1])
        l.append(loss)
        norm_hist.append(np.linalg.norm(grad))
        #print("step:",k)
        if k % 5 == 0:
            print("norm:", np.linalg.norm(grad))        
            #print("loss:", loss)
        #print("new x:", x)
        #print("result",np.diag(np.exp(x)).dot(A).dot(np.exp(x)))
        newGrad = gradient(x, A)
        if len(s) == m:
            #print("hoge")
            s.popleft()
            y.popleft()
        s.append(d)
        #print("y:", newGrad-grad)
        y.append(newGrad-grad)
        grad = newGrad
        k += 1
    #print(np.diag(np.exp(x)).dot(A).dot(np.exp(x)))
    #print("loss:", loss)
    f1.close()
    f2.close()
    print(np.exp(x))
    print(np.diag(np.exp(x)).dot(A).dot(np.diag(np.exp(x))))
    #print(k)
    return x, l

if __name__ == '__main__':
    m = np.loadtxt('../data/HIC_%s_1000000_%s.txt.gz' % ('k562', 'exp'), skiprows=1)
    trg = copy.copy(m)
    trg = preprocess(trg)
    #trg = np.array([[1.2,0.4],[0.4, 1.2]], float)
    #trg = np.array([[100,100002,3],[100002,1000,0.9],[3,0.9,5]], float)
    #trg = np.array([[10,1.2,3],[1.2,7,0.9],[3,10,0.9]], float)
    #trg = np.array([[1,2],[2,4]], float)
    trg += 1e-3
    #trg = trg / 100

    #trg = trg
    #print(trg)
    n, m = trg.shape
    x = -np.ones(n)
    #x = np.zeros(n)
    #x = np.array([0, 0])
    start = time.time()
    x, l = l_bfgs(x, trg)
    end = time.time()
    print("elapsed time:{} sec.".format(end-start))
    plt.plot(l)
    plt.yscale('log')
    plt.xticks(list(range(0,len(l), 5)))
    plt.savefig("../mat_newton_loss.png")
