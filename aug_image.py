
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from keras.models import Sequential
import cv2
import skimage
import os
from imgaug.imgaug import augmenters as iaa
import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K


# In[2]:

BATCH_SIZE=20


# In[3]:

#wanlu
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    iaa.CropAndPad(percent=(-0.25, 0.25)),
    iaa.Add((-30, 30)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Superpixels(p_replace=0.5, n_segments=64),
    iaa.Dropout(p=(0, 0.2)),
    iaa.Affine(rotate=(-45, 45))
])


# In[ ]:




# In[ ]:




# In[6]:

class image_util:
    def __init__(self, data_dir, biz_label_file_name, photo_biz_file_name):
        self.batch_index = 0
        image_paths = [os.path.join(data_dir,i) for i in os.listdir(data_dir) if i.endswith('.jpg') and not i.startswith("._")]
        self.images = []
        self.labels = []
        one_hot = self.read_csv_one_hot(biz_label_file_name)
        photo_biz = self.photo_to_biz_id(photo_biz_file_name)
        counter = 0
        for path in image_paths[:10000]:
            counter += 1
            if counter%100 == 0:
                print("++++++")
                print(counter)
            img = cv2.imread(path)
            if img == None:
                continue
            photo_id = os.path.basename(path).split(".")[0]
            self.labels.append(one_hot[photo_biz[photo_id]])
            img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
            self.images.append(img)
        self.labels = np.asarray(self.labels)
        self.images = np.asarray(self.images)
        with open('train_images.npy','w') as f:
            np.save(f, self.images)
        with open('train_labels.npy','w') as f:
            np.save(f, self.labels)
        print(self.images.shape)
        print(self.labels.shape)

    def next_batch(self, batch_size):
        if batch_size + self.batch_index < self.images.shape[0]:
            imgs = self.images[self.batch_index:batch_size + self.batch_index,:,:,:]
            labels = self.labels[self.batch_index:batch_size + self.batch_index,:]
            return imgs, labels
#             return imgs, np.zeros((10,1000))
        else:
            end_len = self.images.shape[0]-self.batch_index
            start_len = batch_size - (self.images.shape[0] - end_len)
            imgs = np.concatenate((self.images[-end_len:,:,:,:],self.images[0:start_len,:,:,:]))
            labels = np.concatenate((self.labels[-end_len:,:],self.labels[0:start_len,:]))
            return imgs, labels
#             return imgs, np.zeros((10,1000))

    def read_csv_one_hot(self, file_name):
        with open(file_name,"r") as f:
            lines = f.readlines()[1:]
        biz_id_to_label = {}
        for line in lines:
            try:
                biz_id_to_label[line.split(",")[0]] = np.zeros(9)
                for label in line.split(",")[1].rstrip().split(' '):
                    biz_id_to_label[line.split(",")[0]][int(label)]=1
            except:
                if not line.split(",")[1].rstrip():
                    continue
        return biz_id_to_label

    def photo_to_biz_id(self, file_name):
        with open(file_name,"r") as f:
            lines = f.readlines()[1:]
        photo_to_biz = {}
        for line in lines:
            photo_to_biz[line.split(",")[0]] = line.split(",")[1].rstrip()
        return photo_to_biz



# In[ ]:

a = image_util('/home/rendaxuan/Documents/workspace/4032/train_photos', '/home/rendaxuan/Documents/workspace/4032/train.csv', '/home/rendaxuan/Documents/workspace/4032/train_photo_to_biz_ids.csv')


# In[ ]:

from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from custom_layers import Scale

def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter



# In[ ]:

weights_path = 'densenet121_weights_tf.h5'
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
sgd = SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
output = model.output
output = Dense(output_dim=9, activation='sigmoid')(output)
new_model = Model(model.input, output)
new_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
for batch_idx in range(1000):
    print("Batch number:"+str(batch_idx))
    images,labels = a.next_batch(BATCH_SIZE)
#     images = np.expand_dims(images,0)
    images_aug = seq.augment_images(images)
    images_aug[:,:,0] = (images_aug[:,:,0] - 103.94) * 0.017
    images_aug[:,:,1] = (images_aug[:,:,1] - 116.78) * 0.017
    images_aug[:,:,2] = (images_aug[:,:,2] - 123.68) * 0.017
#     new_model.fit(images_aug, labels, batch_size=BATCH_SIZE, epochs=100)
    new_model.fit(a.images, a.labels, batch_size=BATCH_SIZE, epochs=100)
    new_model.save(str(batch_idx)+'_model.h5')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



