from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

def all_binary_possibilities(num_inputs):
    if num_inputs == 0:
        return [[]]

    res = []
    for possibility in all_binary_possibilities(num_inputs - 1):
        res += [[1.] + possibility, [0.] + possibility]

    return res
    
def _XOR(a, b):
    return 1.*((a or b) and not (a and b))

def XOR_dataset(num_inputs):
    if num_inputs < 2:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[_XOR(x[0], x[1])] for x in x_data]
    return np.array(x_data), np.array(y_data)
    
def XOR_of_XORs_dataset(num_inputs):
    if num_inputs < 4:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[_XOR(_XOR(x[0], x[1]), _XOR(x[2], x[3]))]  for x in x_data]
    return np.array(x_data), np.array(y_data)

def NXOR_dataset(num_inputs):
    if num_inputs < 2:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[1.-_XOR(x[0], x[1])] for x in x_data]
    return np.array(x_data), np.array(y_data)

def X0_dataset(num_inputs):
    if num_inputs < 1:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[x[0]] for x in x_data]
    return np.array(x_data), np.array(y_data)

def NOTX0_dataset(num_inputs):
    if num_inputs < 1:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[1.-x[0]] for x in x_data]
    return np.array(x_data), np.array(y_data)

def X0NOTX1_dataset(num_inputs):
    if num_inputs < 1:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[x[0] * (1.-x[1])] for x in x_data]
    return np.array(x_data), np.array(y_data)

def NOTX0NOTX1_dataset(num_inputs):
    if num_inputs < 1:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[1. - x[0] * (1.-x[1])] for x in x_data]
    return np.array(x_data), np.array(y_data)

def AND_dataset(num_inputs):
    if num_inputs < 2:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[x[0]*x[1]] for x in x_data]
    return np.array(x_data), np.array(y_data)

def OR_dataset(num_inputs):
    if num_inputs < 2:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[1.*(x[0] or x[1])] for x in x_data]
    return np.array(x_data), np.array(y_data)

def NAND_dataset(num_inputs):
    if num_inputs < 2:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[1.-x[0]*x[1]] for x in x_data]
    return np.array(x_data), np.array(y_data)

def NOR_dataset(num_inputs):
    if num_inputs < 2:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[1.-(x[0] or x[1])] for x in x_data]
    return np.array(x_data), np.array(y_data)

def parity_dataset(num_inputs, num_to_keep=None):
    if num_to_keep is None:
        num_to_keep = num_inputs
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[np.mod(np.sum(x[:num_to_keep]), 2)] for x in x_data]
    return np.array(x_data), np.array(y_data)

def XAO_dataset(num_inputs):
    if num_inputs < 4:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[_XOR(x[0]*x[1], 1*(x[2] or x[3]))]  for x in x_data]
    return np.array(x_data), np.array(y_data)

def ANDXORS_dataset(num_inputs):
    if num_inputs < 4:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[(_XOR(x[0], x[1]) * _XOR(x[2], x[3]))]  for x in x_data]
    return np.array(x_data), np.array(y_data)

# multivariate
def IDENTITY_dataset(num_inputs):
    x_data = all_binary_possibilities(num_inputs)
    return np.array(x_data), np.array(x_data)

def LOO_PARITY_dataset(num_inputs):
    if num_inputs < 3:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[sum(x[:i] + x[i+1:]) % 2 for i in range(len(x))] for x in x_data]
    return np.array(x_data), np.array(y_data)

def PAIR_AND_dataset(num_inputs): 
    if num_inputs < 3:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[x[i]*x[i+1] for i in range(len(x)-1)] + [x[-1] * x[0]] for x in x_data]
    return np.array(x_data), np.array(y_data)

def PAIR_XOR_dataset(num_inputs): 
    if num_inputs < 3:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[_XOR(x[i], [i+1]) for i in range(len(x)-1)] + [_XOR(x[-1], x[0])] for x in x_data]
    return np.array(x_data), np.array(y_data)

def MIX1_dataset(num_inputs): 
    """[ANDXORS, AND(X0,X1), AND(X2,X3), XOR(X0, X1), XOR(X2, X3)]"""
    if num_inputs < 5:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[_XOR(x[0], x[1]) * _XOR(x[2], x[3]), x[0]*x[1], x[2]*x[3], _XOR(x[0],x[1]), _XOR(x[2],x[3])] for x in x_data]
    if num_inputs > 5:
        y_data = np.concatenate([y_data, np.zeros([len(y_data), num_inputs - 5])], axis=-1)
    return np.array(x_data), np.array(y_data)

def MIX2_dataset(num_inputs): 
    """[XAO, XAO(X2, X3, X0, X1), NAND(X0, X1), NAND(X1, X2), NAND(X2, X3)"""
    if num_inputs < 5:
        raise ValueError("Too few inputs")
    x_data = all_binary_possibilities(num_inputs)
    y_data = [[_XOR(x[0]*x[1], 1*(x[2] or x[3])), _XOR(x[2]*x[3], 1*(x[0] or x[1])), 1.-x[0]*x[1], 1.-x[1]*x[2], 1.-x[2]*x[3]] for x in x_data]
    if num_inputs > 5:
        y_data = np.concatenate([y_data, np.zeros([len(y_data), num_inputs - 5])], axis=-1)
    return np.array(x_data), np.array(y_data)
