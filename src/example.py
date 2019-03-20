import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from matrix_balancing import matrix_balancing, preprocess, sinkhorn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", help = "Target matrix data")
    parser.add_argument("filetype", help="File type: hic or csv", type=str)
    parser.add_argument("output", help="Graph between loss and step num", type=str)
    parser.add_argument("balanced", help="Balanced matrix data", default=True, type=str)
    parser.add_argument("--algorithm", help="Algorithm for gradient descent:lbfgs or newton", default="lbfgs", type=str)
    parser.add_argument("--objective", help="Objective function:barrier or capacity", default="barrier", type=str)
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
    result, l = matrix_balancing(trg, algorithm=args.algorithm, objective=args.objective, truncation=args.truncation)
    #result, l = matrix_balancing(trg, algorithm=args.algorithm, objective=args.objective, truncation=args.truncation)
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