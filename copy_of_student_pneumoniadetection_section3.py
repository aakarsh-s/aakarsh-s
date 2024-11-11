# -*- coding: utf-8 -*-

import random

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D
from keras.regularizers import l2
import tensorflow.keras.optimizers as optimizers
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121

from imgaug import augmenters
def augment(data, augmenter):
  if len(data.shape) == 3:
    return augmenter.augment_image(data)
  if len(data.shape) == 4:
    return augmenter.augment_images(data)

def rotate(data, rotate):
  fun = augmenters.Affine(rotate = rotate)
  return augment(data, fun)

def shear(data, shear):
  fun = augmenters.Affine(shear = shear)
  return augment(data, fun)

def scale(data, scale):
  fun = augmenters.Affine(scale = scale)
  return augment(data, fun)

def flip_left_right(data, prob):
  fun = augmenters.Fliplr(p = prob)
  return augment(data, fun)

def flip_up_down(data, prob):
  fun = augmenters.Flipud(p = prob)
  return augment(data, fun)

def remove_color(data, channel):
  new_data = data.copy()
  if len(data.shape) == 3:
    new_data[:,:,channel] = 0
    return new_data
  if len(data.shape) == 4:
    new_data[:,:,:,channel] = 0
    return new_data

class pkg:
  #### DOWNLOADING AND LOADING DATA
  def get_metadata(metadata_path, which_splits = ['train', 'test']):
    '''returns metadata dataframe which contains columns of:
       * index: index of data into numpy data
       * class: class of image
       * split: which dataset split is this a part of?
    '''
    metadata = pd.read_csv(metadata_path)
    keep_idx = metadata['split'].isin(which_splits)
    return metadata[keep_idx]

  def get_data_split(split_name, flatten, all_data, metadata, image_shape):
    '''
    returns images (data), labels from folder of format [image_folder]/[split_name]/[class_name]/
    flattens if flatten option is True
    '''
    sub_df = metadata[metadata['split'].isin([split_name])]
    index  = sub_df['index'].values
    labels = sub_df['class'].values
    data = all_data[index,:]
    if flatten:
      data = data.reshape([-1, np.product(image_shape)])
    return data, labels

  def get_train_data(flatten, all_data, metadata, image_shape):
    return get_data_split('train', flatten, all_data, metadata, image_shape)

  def get_test_data(flatten, all_data, metadata, image_shape):
    return get_data_split('test', flatten, all_data, metadata, image_shape)

  def get_field_data(flatten, all_data, metadata, image_shape):
    field_data, field_labels = get_data_split('field', flatten, all_data, metadata, image_shape)
    field_data[:,:,:,2] = field_data[:,:,:,0]
    field_data[:,:,:,1] = field_data[:,:,:,0]

    rand = random.uniform(-1, 1)

    for i in range(len(field_data)):
      image = field_data[i]

      if abs(rand) < 0.5:
        image = rotate(image, rotate = rand * 40)
      elif abs(rand) < 0.8:
        image = shear(image, shear = rand*40)
      field_data[i] = image
    return field_data, field_labels

class helpers:
  def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
    '''
    if data is a single image, display that image

    if data is a 4d stack of images, display that image
    '''
    num_dims   = len(data.shape)
    num_labels = len(labels)

    if num_dims == 1:
      data = data.reshape(target_shape)
    if num_dims == 2:
      data = data.reshape(np.vstack[-1, image_shape])
    num_dims   = len(data.shape)

    if num_dims == 3:
      if num_labels > 1:
        print('Multiple labels does not make sense for single image.')
        return

      label = labels
      if num_labels == 0:
        label = ''
      image = data

    if num_dims == 4:
      image = data[index, :]
      label = labels[index]

    print('Label: %s'%label)
    plt.imshow(image)
    plt.show()

  def combine_data(data_list, labels_list):
    return np.concatenate(data_list, axis = 0), np.concatenate(labels_list, axis = 0)

  def plot_acc(history, ax = None, xlabel = 'Epoch #'):
    history = history.history
    history.update({'epoch':list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)

    best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
    ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')
    ax.legend(loc = 4)
    ax.set_ylim([0.4, 1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')

    plt.show()

class models:
  def DenseClassifier(hidden_layer_sizes, nn_params):
    model = Sequential()
    model.add(Flatten(input_shape = nn_params['input_shape']))
    model.add(Dropout(0.5))

    for ilayer in hidden_layer_sizes:
      model.add(Dense(ilayer, activation = 'relu'))
      model.add(Dropout(0.5))

    model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))
    model.compile(loss=nn_params['loss'],
                  optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                  metrics=['accuracy'])
    return model

  def CNNClassifier(num_hidden_layers, nn_params):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding = 'same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(num_hidden_layers-1):
        model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 64, activation = 'relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))

    opt = keras.optimizers.legacy.RMSprop(learning_rate=1e-5, decay=1e-6)

    model.compile(loss=nn_params['loss'],
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

  def TransferClassifier(name, nn_params, trainable = False):
    expert_dict = {'VGG16': VGG16,
                   'VGG19': VGG19,
                   'ResNet50':ResNet50,
                   'DenseNet121':DenseNet121}

    expert_conv = expert_dict[name](weights = 'imagenet',
                                              include_top = False,
                                              input_shape = nn_params['input_shape'])
    for layer in expert_conv.layers:
      layer.trainable = trainable

    expert_model = Sequential()
    expert_model.add(expert_conv)
    expert_model.add(GlobalAveragePooling2D())

    expert_model.add(Dense(128, activation = 'relu'))

    expert_model.add(Dense(64, activation = 'relu'))

    expert_model.add(Dense(nn_params['output_neurons'], activation = nn_params['output_activation']))

    expert_model.compile(loss = nn_params['loss'],
                  optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    return expert_model

metadata_url         = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/metadata.csv"
image_data_url       = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/image_data.npy'
image_data_path      = './image_data.npy'
metadata_path        = './metadata.csv'
image_shape          = (64, 64, 3)

nn_params = {}
nn_params['input_shape']       = image_shape
nn_params['output_neurons']    = 1
nn_params['loss']              = 'binary_crossentropy'
nn_params['output_activation'] = 'sigmoid'


!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/metadata.csv"
!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/image_data.npy"


_all_data = np.load('image_data.npy')
_metadata = pkg.get_metadata(metadata_path, ['train','test','field'])

get_data_split = pkg.get_data_split
get_metadata    = lambda :                 pkg.get_metadata(metadata_path, ['train','test'])
get_train_data  = lambda flatten = False : pkg.get_train_data(flatten = flatten, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
get_test_data   = lambda flatten = False : pkg.get_test_data(flatten = flatten, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
get_field_data  = lambda flatten = False : pkg.get_field_data(flatten = flatten, all_data = _all_data, metadata = _metadata, image_shape = image_shape)

plot_one_image = lambda data, labels = [], index = None: helpers.plot_one_image(data = data, labels = labels, index = index, image_shape = image_shape);
plot_acc       = lambda history: helpers.plot_acc(history)

combine_data           = helpers.combine_data;

DenseClassifier     = lambda hidden_layer_sizes: models.DenseClassifier(hidden_layer_sizes = hidden_layer_sizes, nn_params = nn_params);
CNNClassifier       = lambda num_hidden_layers: models.CNNClassifier(num_hidden_layers, nn_params = nn_params);
TransferClassifier  = lambda name: models.TransferClassifier(name = name, nn_params = nn_params);

monitor = ModelCheckpoint('./model.keras', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')


X_train, y_train = get_train_data()
X_test, y_test   = get_test_data()

cnn = CNNClassifier(num_hidden_layers = 3)
cnn_history = cnn.fit(X_train, y_train, epochs = 50, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])
plot_acc(cnn_history)


X_field, y_field = get_field_data()


y_pred = (cnn.predict(X_field) > 0.5)
accuracy_score(y_field, y_pred)

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()
X_field, y_field = get_field_data()

average_accuracy = 0.0
for i in range(5):
  cnn_temp = CNNClassifier(5)
  cnn_temp.fit(X_train, y_train, epochs = 50, validation_data = (X_test, y_test), shuffle = True, callbacks = (monitor))

  y_pred = (cnn_temp.predict(X_field) > 0.5)
  accuracy = accuracy_score(y_field, y_pred)
  print('Accuracy on this run: %0.2f' % accuracy)

  average_accuracy +=accuracy / 5.0
print('Average accuracy: ', average_accuracy)

print("TEST DATA")
for i in range(2):
  plot_one_image(X_test, y_test, i)

print("FIELD DATA")
for i in range(2):
  plot_one_image(X_field, y_field, i)

print("TRAIN DATA")
for i in range(2):
  plot_one_image(X_train, y_train, i)
    
image = X_train[0]
plot_one_image(image)
new_image = rotate(image, rotate = 40)
plot_one_image(new_image)


image = X_train[0]
plot_one_image(image)
new_image = rotate(new_image, rotate = 40)
new_image = scale(new_image, scale = 1.5)

plot_one_image(new_image)


train_data_rotated_10 = rotate(X_train, rotate=10)



from imgaug import augmenters as iaa

train_data_rotated_45 = rotate(X_train, rotate=45)
train_data_sheared_15 = shear(X_train, shear=15)
train_data_scaled_0_8 = scale(X_train, scale=0.8)
train_data_flipped_lr = flip_left_right(X_train, prob=0.5)
train_data_flipped_ud = flip_up_down(X_train, prob=0.5)
train_data_no_green = remove_color(X_train, channel=1)

def add_gaussian_noise(data, mean=0, stddev=0.1):
    aug = iaa.AdditiveGaussianNoise(mean, stddev)
    return augment(data, aug)

def adjust_contrast(data, factor):
    aug = iaa.LinearContrast(factor)
    return augment(data, aug)

def adjust_brightness(data, factor):
    aug = iaa.Multiply((factor, factor))
    return augment(data, aug)

def random_crop(data, crop_size):
    aug = iaa.CropToFixedSize(width=crop_size[0], height=crop_size[1])
    return augment(data, aug)

def adjust_gamma(data, gamma):
    aug = iaa.GammaContrast(gamma)
    return augment(data, aug)

train_data_gaussian_noise = add_gaussian_noise(X_train, mean=0, stddev=0.05)
train_data_contrast = adjust_contrast(X_train, factor=1.5)
train_data_brightness = adjust_brightness(X_train, factor=1.2)
train_data_cropped = random_crop(X_train, crop_size=(100, 100))
train_data_gamma = adjust_gamma(X_train, gamma=1.5)

all_data, all_labels = combine_data(
    [X_train, train_data_rotated_45, train_data_sheared_15, train_data_scaled_0_8,
     train_data_flipped_lr, train_data_flipped_ud, train_data_no_green,
     train_data_gaussian_noise, train_data_contrast, train_data_brightness,
     train_data_cropped, train_data_gamma],
    [y_train] * 12)


X_train, y_train = get_train_data()
  X_test, y_test   = get_test_data()
  X_field, y_field   = get_field_data()

  average_accuracy = 0.0

  for i in range(5):
    cnn = CNNClassifier(2)
    cnn.fit(all_data, all_labels, epochs = 100, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])
    predictions = (cnn.predict(X_field) > 0.5)
    accuracy = accuracy_score(y_field, predictions)
    print('Accuracy:%0.2f'%accuracy)
    average_accuracy += accuracy

  average_accuracy /= 5.0

  print('Average accuracy: ', average_accuracy)


from tensorflow.keras.optimizers import Adam

learning_rate = 0.00001 

cnn = CNNClassifier(num_hidden_layers=4)
cnn.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

cnn_history = cnn.fit(all_data, all_labels, epochs=15, validation_data=(X_test, y_test), shuffle=True)
predictions = (cnn.predict(X_field) > 0.5)
accuracy = accuracy_score(y_field, predictions)
print('Accuracy: %0.2f' % accuracy)

from google.colab import drive
import tensorflow as tf
drive.mount('/content/gdrive')
save_path = "/content/gdrive/My Drive/cnn_model.keras"

import tensorflow as tf
cnn.save(save_path)

