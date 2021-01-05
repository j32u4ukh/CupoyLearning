import math
import warnings
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


class Scaler(metaclass=ABCMeta):
    def __init__(self):
        self.data = None

    @abstractmethod
    def fit(self, data):
        data = np.array(data)
        self.data = data.copy()
        return data

    @abstractmethod
    def transform(self, data):
        pass

    def fitTransform(self, data):
        self.fit(data)

        return self.transform(data)


class StandardScaler(Scaler):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, data):
        data = super().fit(data)
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError(f"mean: {self.mean}, std: {self.std}")

        return (data - self.mean) / (self.std + 1e-8)

    def fitTransform(self, data):
        self.fit(data)

        return self.transform(data)


def showImage(*args):
    for index, arg in enumerate(args):
        cv2.imshow("img {}".format(index), arg)

    # 按任意建結束
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImages(**kwargs):
    for key in kwargs:
        cv2.imshow("{}".format(key), kwargs[key])

    # 按任意建結束
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def siftDetect(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(img, None)
    return keypoints


def siftDetectAndCompute(img):
    sift = cv2.xfeatures2d_SIFT.create()
    keypoints, features = sift.detectAndCompute(img, None)
    return keypoints, features

