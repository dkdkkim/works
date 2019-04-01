def build_model():
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D
    from keras.layers.normalization import BatchNormalization
    import numpy as np
    # np.random.seed(1000)

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv3D(filters=96, input_shape=(48, 48, 48, 1), kernel_size=(5, 5, 5), strides=(2, 2, 2), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv3D(filters=256, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17))
    model.add(Activation('softmax'))

    model.summary()

    return model

def build_data(train_dir,val_dir):

    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator()
    train_data = datagen.flow_from_directory('train_dir', target_size=(80, 100),
                                             color_mode='grayscale', classes=None, class_mode='categorical',
                                             batch_size=32, interpolation='nearest')
    val_data = datagen.flow_from_directory('val_dir', target_size=(80, 100),
                                            color_mode='grayscale', classes=None, class_mode='categorical',
                                            batch_size=32, interpolation='nearest')

    return train_data,val_data