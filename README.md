# Matrix Balancing Library for square matrix

## Description

Python3 implementation of several matrix balancing algorithms.

L-BFGS algorithm with barrier function as obejective to optimize is the fastest.

## Prerequisite

- Python3
- numpy
- scipy
- matplotlib

## Setup and sample execute

```
move to your workspace
$ git clone https://github.com/fullswing/matrix_balancing.git
$ cd ./matrix_barancing/
$ mkdir result
$ cd src
$ python example.py -h
usage: example.py [-h] [--algorithm ALGORITHM] [--objective OBJECTIVE]
                  [--skiprows SKIPROWS] [--delimiter DELIMITER]
                  [--max_iter MAX_ITER] [--truncation] [--preprocess]
                  [--sinkhorn]
                  matrix filetype output balanced

positional arguments:
  matrix                Target matrix data
  filetype              File type: hic or csv
  output                Graph between loss and step num
  balanced              Balanced matrix data

optional arguments:
  -h, --help            show this help message and exit
  --algorithm ALGORITHM
                        Algorithm for gradient descent:lbfgs or newton
  --objective OBJECTIVE
                        Objective function:barrier or capacity
  --skiprows SKIPROWS   Skip rows
  --delimiter DELIMITER
                        Delimiter
  --max_iter MAX_ITER   Max number of iteration for L-BFGS algorithm
  --truncation          Truncation is activated with this option
  --preprocess          Preprocess the target matrix with this option
  --sinkhorn            Run sinkhorn once and then apply optimization
$ python example.py ../data/hessenberg20.txt csv ../result/barrier_hessenberg_loss_with_truncation.png ../result/barrier_hessenberg20.csv --algorithm lbfgs --truncation  --delimiter , --objective barrier
```

## Result

| Before balancing | After balancing |
|:-----------:|:------------:|
| ![hessenberg](https://github.com/fullswing/matrix_balancing/blob/images/images/hessenberg20.png) | ![balanced hessenberg](https://github.com/fullswing/matrix_balancing/blob/images/images/balanced_hessenberg.png) |