import numpy as np
import math
from keras.layers import Convolution2D
from keras.layers import Input
from utils.dl import computeFeatureSize


def computePaddingSize(in_size, out_size):
    return math.ceil((out_size - in_size + 1) / 2)


def scalePadding(x, output_shape):
    n_dim = len(x.shape)

    if n_dim == 3:
        h_in, w_in = x.shape[0:2]

    elif n_dim == 4:
        h_in, w_in = x.shape[1:3]

    else:
        print("n_dim:", n_dim)
        h_in, w_in = x.shape

    h_out, w_out = output_shape

    h_padding = computePaddingSize(in_size=h_in, out_size=h_out)
    w_padding = computePaddingSize(in_size=w_in, out_size=w_out)

    if n_dim == 3:
        padding_config = ((w_padding, w_padding), (h_padding, h_padding), (0, 0))

    elif n_dim == 4:
        padding_config = ((0, 0), (w_padding, w_padding), (h_padding, h_padding), (0, 0))

    else:
        padding_config = ((w_padding, w_padding), (h_padding, h_padding))

    dst = np.pad(x, padding_config, "constant", constant_values=(0, 0))

    return dst


x = np.random.randint(low=0, high=255, size=(32, 32, 3))
y = scalePadding(x, output_shape=(71, 71))
print(y.shape)
