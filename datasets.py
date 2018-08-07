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
