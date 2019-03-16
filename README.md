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
$ mkdir result | cd src
$ python quasi_newton.py
usage: quasi_newton.py [-h] [--skiprows SKIPROWS] [--delimiter DELIMITER]
                       [--max_iter MAX_ITER] [--truncation] [--preprocess]
                       matrix filetype output balanced

positional arguments:
  matrix                Target matrix data
  filetype              File type: hic or csv
  output                Graph between loss and step num
  balanced              Balanced matrix data

optional arguments:
  -h, --help            show this help message and exit
  --skiprows SKIPROWS   Skip rows
  --delimiter DELIMITER
                        Delimiter
  --max_iter MAX_ITER   Max number of iteration for L-BFGS algorithm
  --truncation          Truncation is activated with this option
  --preprocess          Preprocess the target matrix with this option
$ python quasi_newton.py ../data/hessenberg20.txt csv ../result/hessenberg_exp_loss_with_truncation.png ../result/balanced_hessenberg20.csv --truncation  --delimiter ,
```