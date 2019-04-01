from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding3D
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
import numpy as np
import keras
import tensorflow as tf

model = Sequential()

model.add(Conv3D(filters=96, input_shape=(48, 48, 48, 1), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu',
                 padding='same'))
# model.add(layers.convolutional.ZeroPadding3D((1, 1, 1), input_shape=(48, 48, 48,1)))
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
# model.add(MaxPooling3D((2, 2, 2)))

# model.add(layers.convolutional.ZeroPadding2D((1, 1)))
model.add(Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(BatchNormalization())
model.add(Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

# model.add(layers.convolutional.ZeroPadding2D((1, 1)))
model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(BatchNormalization())
model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

# model.add(layers.convolutional.ZeroPadding2D((1, 1)))
model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(BatchNormalization())
model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
# model.add(MaxPooling3D((2, 2, 2)))

# model.add(layers.convolutional.ZeroPadding2D((1, 1)))
model.add(Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling3D((2, 2, 2)))

# model.add(layers.convolutional.ZeroPadding2D((1, 1)))
model.add(Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling3D((7, 7, 7)))

model.add(Flatten())
model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu', kernel_initializer='he_normal'))
model.add(Dense(2, kernel_regularizer=regularizers.l2(0.001), activation='softmax'))

model.summary()