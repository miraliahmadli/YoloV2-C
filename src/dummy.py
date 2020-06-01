import numpy as np
import ctypes
from ctypes import *

result = np.array(
    [
        [1., 2., 3.],
        [-1., -2., -3.],
        [-9.49481346, -9.13493229, -9.13493229],
        [-4, 3, 4]
    ], dtype = c_float
)

bias = np.array(
    [ 9., 8., 7], dtype = c_float
)

mylib = cdll.LoadLibrary('./cuda_lib.so')

func = mylib.add_bias
func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
res_p = result.ctypes.data_as(POINTER(c_float))
bias_p = bias.ctypes.data_as(POINTER(c_float))

func(res_p, bias_p, 4, 3)

print(result)
print(bias)
