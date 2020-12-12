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
