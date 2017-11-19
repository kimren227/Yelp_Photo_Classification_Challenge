from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import random
import models
import numpy as np
import os
import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


BATCH_SIZE = 4
DATA_DIR = './data'
def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def build_model():
    input_placeholder = Input(shape=(224,224,3),name="placeholder")
    q_input = Input(shape=(224,224,3),name='q_data')
    p_input = Input(shape=(224,224,3),name='p_data')
    n_input = Input(shape=(224,224,3),name='n_data')

    feature_extractor = models.Feature_extractor(input_placeholder)
    shared_module = Model(input_placeholder,feature_extractor)
    positive_item_embedding = shared_module(p_input)
    negative_item_embedding = shared_module(n_input)
    query_item_embedding = shared_module(q_input)

    loss = merge(
            [positive_item_embedding, negative_item_embedding, query_item_embedding],
            mode=bpr_triplet_loss,
            name='loss',
            output_shape=(1, ))

    model = Model(
            input=[p_input, n_input, q_input],
            output=loss)
    model.compile(loss=identity_loss, optimizer=Adam())
    return model

data = np.expand_dims(np.random.randint(1000,size=(224,224,3)),0)

def preprocess_image(path):
    im = cv2.imread(path)
    print(im.shape)
    im = cv2.resize(im, (224,224)).astype(np.float32)
    im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
    im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
    im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
    return im


def get_data(BATCH_SIZE, data_dir):
    classes_dir =[os.path.join(data_dir,d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
    print(os.listdir(data_dir))
    print(classes_dir)
    for i in range(BATCH_SIZE):
        q_p_dir = random.choice(classes_dir)
        n_dir = random.choice(classes_dir)
        print(q_p_dir)
        choice_time = 0
        while n_dir == q_p_dir:
            n_dir = random.choice(classes_dir)
            choice_time += 1
            if choice_time > 10:
                print("Data dir may have a problem")
                raise AssertionError()
        q_image = preprocess_image(os.path.join(q_p_dir,random.choice(os.listdir(q_p_dir))))
        p_image = preprocess_image(os.path.join(q_p_dir,random.choice(os.listdir(q_p_dir))))
        n_image = preprocess_image(os.path.join(n_dir,random.choice(os.listdir(n_dir))))
        q_image = np.expand_dims(q_image,axis=0)
        p_image = np.expand_dims(p_image, axis=0)
        n_image = np.expand_dims(n_image, axis=0)
        if i == 0:
            batch_q_images = q_image
            batch_p_images = p_image
            batch_n_images = n_image
        else:
            batch_q_images = np.concatenate((batch_q_images, q_image),axis=0)
            batch_p_images = np.concatenate((batch_p_images, p_image), axis=0)
            batch_n_images = np.concatenate((batch_n_images, n_image), axis=0)
    print(batch_q_images.shape)
    return (batch_q_images,batch_p_images,batch_n_images)



model = build_model()

# for i in range(100):
q_batch, p_batch, n_batch = get_data(BATCH_SIZE, DATA_DIR)
# *_data.shape = (BATCH_SIZE,224,224,3)
X = {
        'q_data': q_batch,
        'p_data': p_batch,
        'n_data': n_batch
    }

print(model.summary())

num_epochs = 10
for i in range(num_epochs):
    model.fit(X,
                  np.ones((BATCH_SIZE,1)),
                  batch_size=BATCH_SIZE,
                  nb_epoch=1,
                  verbose=1,
                  shuffle=True)
    print('hiuhiuhiu')

