import numpy as np
from sklearn.model_selection import train_test_split
import keras
from utils.opencv import StandardScaler


seed = 7
np.random.seed(seed)

n_class = 10

# x_train.shape = (50000, 32, 32, 3)
x_train = np.random.randint(low=0, high=255, size=(50000, 32, 32, 3), dtype=np.int)
# y_train.shape = (50000, 1)
y_train = np.random.randint(low=1, high=n_class, size=(50000, 1), dtype=np.int)

x_test = np.random.randint(low=0, high=255, size=(10000, 32, 32, 3), dtype=np.int)
y_test = np.random.randint(low=1, high=n_class, size=(10000, 1), dtype=np.int)

scaler = StandardScaler()
x_train_norm = scaler.fitTransform(x_train)
x_test_norm = scaler.transform(x_test)

y_train_onehot = keras.utils.to_categorical(y_train, n_class)
y_test_onehot = keras.utils.to_categorical(y_test, n_class)

# 30%用於測試集，70%用於訓練集
x_train_norm, x_validation_norm, y_train_onehot, y_validation_onehot = train_test_split(x_train_norm, y_train_onehot,
                                                                                        test_size=0.3,
                                                                                        random_state=seed)
