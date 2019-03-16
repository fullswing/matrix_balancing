# Sinkhorn knopp
import numpy as np 
import copy, time
import numpy as np
import copy, time
import sys
import argparse

 
def sinkhorn(A):
    A = np.array(A)
    row, col = A.shape
    scale_row = [1/sum(A[r]) for r in range(row)]
    A = (A.T * scale_row).T
    scale_col = [1/sum(A[:,c]) for c in range(col)]
    A = A * scale_col
    return A

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", help = "Target matrix data")
    parser.add_argument("filetype", help="File type: hic or csv", type=str)
    parser.add_argument("--skiprows", help="Skip rows", default=0, type=int)
    parser.add_argument("--delimiter", help="Delimiter", default="", type=str)
    args = parser.parse_args()
    if args.filetype == "hic":
        mat = np.loadtxt(args.matrix, skiprows=args.skiprows)
    elif args.filetype == "csv":
        mat = np.genfromtxt(args.matrix, delimiter=args.delimiter, skip_header=args.skiprows)
    else:
        print("Error:Invalid file type", file=sys.stderr)
        return
    th = 1e-6
    loss = 1
    step = 0
    l = []
    trg = copy.copy(mat)
    row, col = trg.shape
    s = time.time()
    while loss > th:
        loss = 0
        trg = sinkhorn(trg)
        step += 1
        for r in range(row):
            loss += np.abs(sum(trg[r]) - 1) ** 2
        for c in range(col):
            loss += np.abs(sum(trg[:,c]) - 1) ** 2
        if step % 5 == 0:
            print("step:{} loss:{}".format(step, loss))
    t = time.time()
    print(t-s)


if __name__ == '__main__':
    main()
    