import random

import cv2
import numpy as np
from imgaug import augmenters as iaa
from keras.models import Sequential
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator

from utils.opencv import showImages

"""Working directory: CupoyLearning

『本次練習內容』
學習使用 Keras-ImageDataGenerator 與 Imgaug 做圖像增強

『本次練習目的』
熟悉 Image Augmentation 的實作方法
瞭解如何導入 Imgae Augmentation 到原本 NN 架構中
"""

# region ========== Part1 ==========
# Training Generator
train_generator = ImageDataGenerator(zca_whitening=False,
                                     # rescale=2,
                                     rotation_range=10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=False,
                                     vertical_flip=False)

# Test Generator，只需要 Rescale，不需要其他增強
test_generator = ImageDataGenerator(rescale=1.0 / 255.0)

width = 224
height = 224
batch_size = 4

img = cv2.imread('image/Tano.JPG')

# 改變圖片尺寸
img = cv2.resize(img, (224, 224))

# cv2讀進來是BGR，轉成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_origin = img.copy()
img = np.array(img, dtype=np.float32)

showImages(img_origin=img_origin, img=img)

# 輸入generator要是四維，(224,224,3)變成(4,224,224,3)
img_combine = np.array([img, img, img, img], dtype=np.uint8)
batch_gen = train_generator.flow(img_combine, batch_size=4)
assert next(batch_gen).shape == (batch_size, width, height, 3)

images = next(batch_gen)
images = images.astype(np.uint8)
showImages(origin=img_origin, gen0=images[0], gen1=images[1], gen2=images[2], gen3=images[3])

"""
示範如何導入 ImageDataGenerator 到 Keras 訓練中
"""

# 將路徑給 Generator，自動產生 Label
training_set = train_generator.flow_from_directory('dataset/training_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='categorical')

valid_set = train_generator.flow_from_directory('dataset/valid_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

test_set = test_generator.flow_from_directory('dataset/test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='categorical')

# 訓練
model = Sequential()
model.fit_generator(training_set, steps_per_epoch=250, nb_epoch=25, validation_data=valid_set, validation_steps=63)

# 預測新照片
test_image = image_utils.load_img('dataset/new_images/new_picture.jpg', target_size=(224, 224))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict_on_batch(test_image)

"""練習使用 Imgaug

使用單項增強
"""

images = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)  # 創造一個array size==(5, 224, 224, 3)

# 水平翻轉機率: 1.0
flipper = iaa.Fliplr(1.0)
images[0] = flipper.augment_image(img)

vflipper = iaa.Flipud(0.4)  # 垂直翻轉機率40%
images[1] = vflipper.augment_image(img)

blurer = iaa.GaussianBlur(3.0)
images[2] = blurer.augment_image(img)  # 高斯模糊圖像(sigma of 3.0)

translater = iaa.Affine(translate_px={"x": -16})  # 向左橫移16個像素
images[3] = translater.augment_image(img)

scaler = iaa.Affine(scale={"y": (0.8, 1.2)})  # 縮放照片，區間(0.8-1.2倍)
images[4] = scaler.augment_image(img)

showImages(gen0=images[0], gen1=images[1], gen2=images[2], gen3=images[3], gen4=images[4])


# endregion


# region ========== Part2 ==========
# Sometimes(0.5, ...) 代表每次都有 50% 的機率運用不同的 Augmentation
def sometime(augmentation, p=0.5):
    return iaa.Sometimes(p, augmentation)


class Compose:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, images):
        for aug in self.augmentations:
            images = aug(images)
        return images


class ImgaugAugmentation:
    def __init__(self, augmentations: list, random_order=True):
        # 包裝想運用之圖像強化方式
        self.equential = iaa.Sequential(augmentations, random_order=random_order)

    def __call__(self, images):
        augmented_images = self.equential.augment_images(images)
        return augmented_images


class RandomBrightness:
    """隨機改變亮度 Function to randomly make image brighter or darker
    Parameters
    ----------
    delta: float
        the bound of random.uniform distribution
    """

    def __init__(self, delta=16):
        assert 0 <= delta <= 255
        self.delta = delta

    def __call__(self, images):
        delta = random.uniform(-self.delta, self.delta)
        if random.randint(0, 1):
            images = images + delta
        images = np.clip(images, 0.0, 255.0)
        images = np.int8(images)
        return images


class RandomContrast:
    """隨機改變對比 Function to strengthen or weaken the contrast in each image
    Parameters
    ----------
    lower: float
        lower bound of random.uniform distribution
    upper: float
        upper bound of random.uniform distribution
    """

    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower, "contrast upper must be >= lower."
        assert lower >= 0, "contrast lower must be non-negative."
        self.lower = lower
        self.upper = upper

    def __call__(self, images):
        alpha = random.uniform(self.lower, self.upper)

        if random.randint(0, 1):
            images = images * alpha

        images = np.clip(images, 0.0, 255.0)
        images = np.int8(images)

        return images


class TrainAugmentations:
    def __init__(self):
        self.aug_pipeline = Compose([
            RandomBrightness(16),  # make image brighter or darker
            RandomContrast(0.9, 1.1),  # strengthen or weaken the contrast in each image
            ImgaugAugmentation([
                sometime(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5)),
                sometime(iaa.ContrastNormalization((0.5, 2.0), per_channel=1)),

                # sharpen images
                sometime(iaa.Sharpen(alpha=(0, 0.2), lightness=(0.1, 0.4))),

                # emboss images
                sometime(iaa.Emboss(alpha=(0, 0.3), strength=(0, 0.5)))],
                random_order=True),
        ])

    def __call__(self, image):
        image = self.aug_pipeline(image)
        return image


class MaskAugSequence:
    def __init__(self, sequence):
        self.sequence = sequence

    def __call__(self, image, mask):
        # 用來關閉隨機性
        sequence = self.sequence.to_deterministic()
        image = sequence.augment_image(image)
        mask = sequence.augment_image(mask)
        image, mask = image.astype(np.float32), mask.astype(np.float32)
        return image, mask


trainAugmentations = TrainAugmentations()

img = cv2.imread('image/Tano.JPG')

# 改變圖片尺寸
img = cv2.resize(img, (224, 224))

# cv2讀進來是BGR，轉成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

randomBrightness = RandomBrightness()
random_brightness = randomBrightness(img.copy())

randomContrast = RandomContrast()
random_contrast = randomContrast(img.copy())

train_augmentations = trainAugmentations(img.copy())
showImages(img=img,
           train_augmentations=train_augmentations,
           random_brightness=random_brightness,
           random_contrast=random_contrast)
# endregion
