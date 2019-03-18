# Matrix Balancing using quasi newton optimization

## Description

Python3 implementation of matrix balancing using quasi-newton optimization, basically for square matrix

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
$ python quasi_newton.py -h
usage: quasi_newton.py [-h] [--algorithm ALGORITHM] [--skiprows SKIPROWS]
                       [--delimiter DELIMITER] [--max_iter MAX_ITER]
                       [--truncation] [--preprocess] [--sinkhorn]
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
  --skiprows SKIPROWS   Skip rows
  --delimiter DELIMITER
                        Delimiter
  --max_iter MAX_ITER   Max number of iteration for L-BFGS algorithm
  --truncation          Truncation is activated with this option
  --preprocess          Preprocess the target matrix with this option
  --sinkhorn            Run sinkhorn once and then apply optimization
$ python quasi_newton.py ../data/hessenberg20.txt csv ../result/hessenberg_loss_with_truncation.png ../result/balanced_hessenberg20.csv --truncation  --delimiter ,
```

## Result

| Before balancing | After balancing |
|:-----------:|:------------:|
| ![hessenberg](https://github.com/fullswing/matrix_balancing/blob/images/images/hessenberg20.png) | ![balanced hessenberg](https://github.com/fullswing/matrix_balancing/blob/images/images/balanced_hessenberg.png) |