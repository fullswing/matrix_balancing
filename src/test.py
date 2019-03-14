import numpy as np 
import copy
from quasi_newton import preprocess

def test_preprocess():
    print("case1")
    A = np.array([[],[],[]])
    B = copy.copy(A)
    ans = np.array([[], [], []])
    res = preprocess(B)
    #print(res)
    assert np.array_equal(ans, res)

    print("case2")
    A = np.array([[2, 1],[0, 1], [0, 0]])
    B = copy.copy(A)
    ans = np.array([])
    res = preprocess(B)
    #print(type(res))
    #print(type(ans))
    assert np.count_nonzero(res) == np.count_nonzero(ans)

    print("case3")
    A = np.array([[1, 0, 2, 3], [0, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]])
    B = copy.copy(A)
    ans = np.array([[1, 2, 3], [2, 1, 0], [3, 0, 1]])
    res = preprocess(B)
    assert np.array_equal(ans, res)

if __name__ == '__main__':
    test_preprocess()