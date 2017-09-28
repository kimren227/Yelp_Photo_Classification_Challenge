import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from keras.preprocessing.image import ImageDataGenerator, img_to_array

def PreprocessImage(path, show_img=True):
    # load image
    img = io.imread(path)
    img_name = os.path.basename(path)
    
    # print("Original Image Shape: ", img.shape)

    # crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    resized_img = transform.resize(crop_img, (299, 299))

    # for testing purpose
    # disable show_img when process data
    if show_img:
        plt.imshow(resized_img)

    sample = img_to_array(resized_img) * 256
    # sub mean
    normed_img = (sample - 128.)/128.
    normed_img = normed_img.reshape((1,) + normed_img.shape)

    return normed_img

if __name__ == '__main__':
    # keras ImageDataGenerator
    # https://keras.io/preprocessing/image/
    datagen = ImageDataGenerator(
            rotation_range=10,
            shear_range=0.2,
            zoom_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True)

    # change the file name
    normed_img = PreprocessImage('./o.jpg')

    # generate 10 augmented data
    i = 0
    for batch in datagen.flow(normed_img,
                              batch_size=1,
                              # change the directory
                              save_to_dir='./data_augmentation',
                              save_prefix=img_name,
                              save_format='jpeg'):
        i += 1
        if i > 10:
            break
