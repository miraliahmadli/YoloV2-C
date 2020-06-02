import numpy as np
import ctypes
from ctypes import *

# prev_res = np.array(
#     [
#         [1., 2., 3., 4.],
#         [-1., -2., -3., -4],
#         [10, 100, 1000, 10000],
#         [-10, -100, -1000, -10000]
#     ], dtype = c_float
# )

# expected = [
#     [2, 4],
#     [100, 10000]
# ]
# result = np.zeros((2,2)).astype(c_float)
# ksize = 2

# bias = np.array(
#     [ 9., 8., 7], dtype = c_float
# )

# mylib = cdll.LoadLibrary('./cuda_lib.so')

# # func = mylib.maxpool2d
# # func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t,
# #                     c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]
# # res_p = result.ctypes.data_as(POINTER(c_float))
# # prev_p = prev_res.ctypes.data_as(POINTER(c_float))

# # func(res_p, prev_p, 2, 2, 2, 2, 4, 4, 1)
# func = mylib.maxpool2d
# func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t,
#                     c_size_t, c_size_t, c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]
# res_p = result.ctypes.data_as(POINTER(c_float))
# prev_p = prev_res.ctypes.data_as(POINTER(c_float))

# func(res_p, prev_p, 2, 2, 2, 2, 4, 4, 1, 4, 16)
# print(result)
# print(prev_res)
import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    out_height = int(out_height)
    out_width = int(out_width)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

# def conv_forward(X, W, stride=1, padding=1):
#     h_filter, w_filter, d_filter, n_filters = W.shape
#     n_x, d_x, h_x, w_x = X.shape
#     h_out = (h_x - h_filter + 2 * padding) / stride + 1
#     w_out = (w_x - w_filter + 2 * padding) / stride + 1

#     if not h_out.is_integer() or not w_out.is_integer():
#         raise Exception('Invalid output dimension!')

#     h_out, w_out = int(h_out), int(w_out)

#     X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
#     print("\n\n\n")
#     print(X_col)
#     W_col = W.reshape(-1, n_filters)
#     W_col = W_col.transpose()

#     out = W_col @ X_col
#     out = out.reshape(n_filters, h_out, w_out, n_x)
#     out = out.transpose(3, 0, 1, 2)

#     return out

def conv_forward(X, W, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    print("\n\n\n")
    print(X_col)
    W_col = W.reshape(n_filters, -1)
    print(W_col.shape)

    out = W_col @ X_col
    print(out.shape)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    return out

prev_res = np.array(
    [
        [
            [
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.]
            ],
            [
                [17., 18., 19., 20.],
                [21.,22., 23., 24.],
                [25., 26., 27., 28.],
                [29., 30., 31., 32.]
            ],
            [
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.]
            ]
        ]
    ]
)
prev_res1 = np.array(
    [
        [
            [
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.]
            ],
        ]
    ]
)

filt = np.array(
    [
        [
            [
                [1., 2.],
                [3., 4.],
            ],
        ]
    ]
)
filt1 = np.array(
    [
        [
            [
                [1., 2.],
                [3., 4.],
            ],
            [
                [5., 6.],
                [7., 8.],
            ],
            [
                [9., 10.],
                [11., 12.],
            ],
        ],
        [
            [
                [13., 14.],
                [15., 16.],
            ],
            [
                [17., 18.],
                [19., 20.],
            ],
            [
                [21., 22.],
                [23., 24.],
            ],
        ]
    ]
)

print(prev_res.shape)
# print(filt1.shape)

# res = conv_forward(prev_res, filt1, stride=1, padding=0)
# print(res)
# print(res.shape)
# print(res[0].shape)
# res = res.transpose(0, 2, 3, 1)


# # First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
# # n h w c
# X = X.transpose(0, 3, 1, 2)
# X_reshaped = X.reshape(n * d, 1, h, w)

# X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

# max_idx = np.argmax(X_col, axis=0)

# out = X_col[max_idx, range(max_idx.size)]
# out = out.reshape(h_out, w_out, n, d)
# out = out.transpose(2, 0, 1, 3)


        # n, h, w, d = self.in_node.out_shape
        # pad = 0
        # if self.padding:
        #     # ((s-1) * x + k -s)/ 2
        #     pad = self.ksize[1] - 1
        # X = self.in_node.result.transpose(0, 3, 1, 2)
        # X_reshaped = X.reshape(n * d, 1, h, w)

        # if self.strides[1] == 1:
        #     X_col = im2col_indices(X_reshaped, self.ksize[1], self.ksize[2], pad, self.strides[1])
        # else:
        #     X_col = im2col_indices(X_reshaped, self.ksize[1], self.ksize[2], 2*(pad//2), self.strides[1])

        # max_idx = np.argmax(X_col, axis=0)
        # _, h_out, w_out, _ = self.out_shape
        # out = X_col[max_idx, range(max_idx.size)]
        # out = out.reshape(h_out, w_out, n, d)
        # out = out.transpose(2, 0, 1, 3)
        # self.result = out
        # print(self.result.shape)
        # print(self.out_shape)

# LRELU

# self.result = np.copy(self.in_node.result)
# result = self.result.astype(c_float)
# b, h, w, c = self.out_shape

# func = mylib.leaky_relu
# func.argtypes = [POINTER(c_float), c_size_t]
# res_p = result.ctypes.data_as(POINTER(c_float))
# func(res_p, b*h*w*c)
# self.result = result.astype("float64")
# X = prev_res
# n, d, h, w = prev_res.shape
# size = 2
# stride = 2
# X_reshaped = X.reshape(n * d, 1, h, w)

# # The result will be 4x9800
# # Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
# X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

# # Next, at each possible patch location, i.e. at each column, we're taking the max index
# max_idx = np.argmax(X_col, axis=0)

# # Finally, we get all the max value at each column
# # The result will be 1x9800
# out = X_col[max_idx, range(max_idx.size)]

# # Reshape to the output size: 14x14x5x10
# out = out.reshape(2, 2, n, d)
# print(out)
# print("\n\n\n")

# # Transpose to get 5x10x14x14 output
# out = out.transpose(2, 3, 0, 1)
# print(out)

prev_res = np.array(
    [
        [1., 2., 3., 4.],
        [-1., -2., -3., -4],
        [10., 100., 1000., 10000.],
        [-10., -100., -1000., -10000.]
    ], dtype = c_double
)

print(prev_res.shape)
print(prev_res)

mylib = cdll.LoadLibrary('./cuda_lib.so')

# func = mylib.maxpool2d
# func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t,
#                     c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]
# res_p = result.ctypes.data_as(POINTER(c_float))
# prev_p = prev_res.ctypes.data_as(POINTER(c_float))

# func(res_p, prev_p, 2, 2, 2, 2, 4, 4, 1)
result = np.zeros((1, 4))
result = result.astype(c_double)
func = mylib.maxpool
func.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t, c_size_t]
res_p = result.ctypes.data_as(POINTER(c_double))
prev_p = prev_res.ctypes.data_as(POINTER(c_double))

func(res_p, prev_p, 4, 4)
print(result)
