def build_model():
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

    model.add(
        Conv3D(filters=96, input_shape=(48, 48, 48, 1), kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu',
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
    # model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    # model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling3D((2, 2, 2)))

    # model.add(layers.convolutional.ZeroPadding2D((1, 1)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D((2, 2, 2)))

    # model.add(layers.convolutional.ZeroPadding2D((1, 1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    # model.add(MaxPooling3D((2, 2, 2)))

    # model.add(layers.convolutional.ZeroPadding2D((1, 1)))d
    model.add(Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D((4, 4, 4)))

    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(2, kernel_regularizer=regularizers.l2(0.001), activation='softmax'))

    model.summary()

    # # 1st Convolutional Layer
    # model.add(Conv3D(filters=96, input_shape=(48, 48, 48, 1), kernel_size=(5, 5, 5), strides=(2, 2, 2), padding='valid'))
    # model.add(Activation('relu'))
    # # Max Pooling
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid'))
    #
    # # 2nd Convolutional Layer
    # model.add(Conv3D(filters=256, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='valid'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid'))
    #
    # # 3rd Convolutional Layer
    # model.add(Conv3D(filters=384, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='valid'))
    # model.add(Activation('relu'))
    #
    # # 4th Convolutional Layer
    # model.add(Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid'))
    # model.add(Activation('relu'))
    #
    # # 5th Convolutional Layer
    # model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid'))
    # model.add(Activation('relu'))
    #
    # # Max Pooling
    # model.add(MaxPooling3D(pool_size=(8, 8, 8), strides=(1, 1, 1), padding='valid'))
    #
    # # Passing it to a Fully Connected layer
    # model.add(Flatten())
    # # # 1st Fully Connected Layer
    # # model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    # # model.add(Activation('relu'))
    # # # Add Dropout to prevent overfitting
    # # model.add(Dropout(0.4))
    #
    # # 2nd Fully Connected Layer
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    #
    # # 3rd Fully Connected Layer
    # model.add(Dense(1000))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    #
    # # Output Layer
    # model.add(Dense(2))
    # model.add(Activation('softmax'))

    # model.summary()


    return model

def build_model_2():
    from keras import regularizers
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D
    from keras.layers.normalization import BatchNormalization


    model = Sequential()

    model.add(Conv3D(filters=96, input_shape=(48, 48, 48, 1), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu',padding='same'))
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

    return model

def build_resnet():
    def skip_connect(input, n_ch):
        x1 = BatchNormalization()(input)
        x1 = layers.Activation('relu')(x1)
        x1 = Conv2D(n_ch / 2, (1, 1), kernel_initializer='he_normal')(x1)
        x1 = layers.convolutional.ZeroPadding2D((1, 1))(x1)
        x1 = BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = Conv2D(n_ch, (3, 3), kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = Conv2D(n_ch / 2, (1, 1), kernel_initializer='he_normal')(x1)

        x2 = BatchNormalization()(input)
        x2 = layers.Activation('relu')(x2)
        x2 = Conv2D(n_ch / 2, (1, 1), kernel_initializer='he_normal')(x2)

        output = concatenate(inputs=[x2, x1])
        return output

    input_dat = Input(shape=(128, 128, 3))

    first_layer = layers.convolutional.ZeroPadding2D((1, 1))(input_dat)
    first_layer = BatchNormalization()(first_layer)
    first_layer = Conv2D(64, (3, 3), kernel_initializer='he_normal')(first_layer)
    # first_layer = layers.Activation('relu')(first_layer)
    # first_layer = layers.MaxPooling2D((2,2))(first_layer)

    x1 = skip_connect(first_layer, 64)
    x1 = skip_connect(x1, 64)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    x2 = skip_connect(x1, 128)
    x2 = skip_connect(x2, 128)
    # x2 = skip_connect(x2, 128)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x2_2 = skip_connect(x2, 256)
    # x2_2 = skip_connect(x2_2, 256)
    x2_2 = skip_connect(x2_2, 256)
    x2_2 = layers.MaxPooling2D((2, 2))(x2_2)

    x3 = skip_connect(x2_2, 512)
    x3 = skip_connect(x3, 512)
    x3 = skip_connect(x3, 512)
    x3 = skip_connect(x3, 512)
    x3 = layers.MaxPooling2D((2, 2))(x2_2)

    x4 = skip_connect(x3, 1024)
    x4 = skip_connect(x4, 1024)
    x4 = skip_connect(x4, 1024)
    x4 = skip_connect(x4, 1024)

    output = layers.AveragePooling2D((8, 8))(x4)
    output = layers.Flatten()(output)
    output = layers.Dense(200, kernel_regularizer=regularizers.l2(0.001), activation='softmax',
                          kernel_initializer='he_normal')(output)
    # output = layers.Dense(200,activation='relu', kernel_initializer='he_normal')(output)

    model = models.Model(inputs=input_dat, outputs=output)

def build_data(train_dir,val_dir):
    from glob import glob
    import numpy as np
    from sklearn.utils import shuffle

    train_dir = '/home/dkkim/data/train'
    val_dir = '/home/dkkim/data/valid'


    ## train

    train_path_0 = glob(train_dir + '/class0/*.npy')
    train_path_1 = glob(train_dir + '/class1/*.npy')



    train_X = np.empty((len(train_path_0)+len(train_path_1), 48, 48, 48))
    train_Y = np.zeros((len(train_path_0)+len(train_path_1), 2))

    for pathIdx, path in enumerate(train_path_0):
        train_X[pathIdx] = np.load(path)
        train_Y[pathIdx,0] = 1

    for pathIdx, path in enumerate(train_path_1):
        train_X[pathIdx + len(train_path_0)] = np.load(path)
        train_Y[pathIdx + len(train_path_0),1] = 1

    train_X = np.expand_dims(train_X, axis=4)

    tX,tY = train_X, train_Y
    # tX, tY = shuffle(train_X, train_Y, random_state=0)

    ## valid

    val_path_0 = glob(val_dir + '/class0/*.npy')
    val_path_1 = glob(val_dir + '/class1/*.npy')

    val_X = np.empty((len(val_path_0) + len(val_path_1), 48, 48, 48))
    val_Y = np.zeros((len(val_path_0) + len(val_path_1), 2))

    for pathIdx, path in enumerate(val_path_0):
        val_X[pathIdx] = np.load(path)
        val_Y[pathIdx, 0] = 1

    for pathIdx, path in enumerate(val_path_1):
        val_X[pathIdx + len(val_path_0)] = np.load(path)
        val_Y[pathIdx + len(val_path_0), 1] = 1

    val_X = np.expand_dims(val_X, axis=4)

    # vX, vY = shuffle(val_X, val_Y, random_state=0)
    vX,vY = val_X, val_Y

    return tX,tY,vX,vY





    from keras.preprocessing.image import ImageDataGenerator

    # datagen = ImageDataGenerator()
    # train_data = datagen.flow_from_directory(train_dir, target_size=(80, 100),
    #                                          color_mode='grayscale', classes=None, class_mode='categorical',
    #                                          batch_size=32, interpolation='nearest')
    # val_data = datagen.flow_from_directory(val_dir, target_size=(80, 100),
    #                                        color_mode='grayscale', classes=None, class_mode='categorical',
    #                                        batch_size=32, interpolation='nearest')
    #
    # return train_data, val_data

def training_model(model,train_x,train_y,valid_x,valid_y,log_data_dir,log_name,epoch,batch,lr,gpus):

    import os
    import csv
    import math
    import keras
    from keras.callbacks import ModelCheckpoint,LearningRateScheduler,Callback
    from keras import optimizers
    import math
    from keras.utils.training_utils import multi_gpu_model
    import tensorflow as tf

    model = multi_gpu_model(model, gpus = gpus)

    # learning rate log
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    #
    # # learning rate decay
    # def step_decay(epoch):
    #     initial_lrate = lr
    #     drop = 0.5
    #     epochs_drop = 10
    #     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)) # lr = lr0 * drop^floor(epoch / epochs_drop)
    #     return lrate
    #
    # def exp_decay(epoch):
    #     initial_lrate = lr
    #     k = 0.1
    #     lrate = initial_lrate * math.exp(-k * epoch)
    #     return lrate
    #
    #
    # # Loss history
    # class LossHistory(Callback):
    #     def on_train_begin(self, logs={}):
    #        self.losses = []
    #        self.lr = []
    #
    #     def on_epoch_end(self, batch, logs={}):
    #        self.losses.append(logs.get("loss"))
    #        self.lr.append(step_decay(len(self.losses)))
    #
    # loss_history = LossHistory()
    #
    optimizer = optimizers.Adam(lr=lr)
    lr_metric = get_lr_metric(optimizer)

    # compile model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy",lr_metric])
    # lrate = LearningRateScheduler(step_decay) ## put into callback_list

    # check point
    checkpoint_dir = os.path.join(log_data_dir, log_name)  ################ version
    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_acc:.2f}.hdf5')# lr = lr0 * drop^floor(epoch / epochs_drop)
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=False, save_weights_only=False,
                                 mode='auto', period=1)


    # training model

    callbacks_list = [checkpoint]
    # callbacks_list = [checkpoint, lrate]

    model_mgpu = multi_gpu_model(model, gpus=gpus)

    model_mgpu.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])


    history = model_mgpu.fit(train_x,train_y, callbacks = callbacks_list, epochs=epoch,batch_size=batch, shuffle=True, validation_data=(valid_x,valid_y),verbose=1)

    # history = model.fit_generator(train_x,train_y, steps_per_epoch=2000, epochs=50,batch_size=100, validation_data=(valid_x,valid_y),validation_steps=800)

    # save weights
    model_weight = os.path.join(log_data_dir, '%s.h5'%log_name)
    model.save(model_weight)

    # save structure
    model_arch = os.path.join(log_data_dir, '%s.json'%log_name)
    with open(model_arch, 'w') as f:
        f.write(model.to_json())

    # save log
    dict = history.history

    csv_path = os.path.join(log_data_dir, '%s.csv'%log_name)

    w = csv.writer(open(csv_path, "w"))
    for key, val in dict.items():
        w.writerow([key, val])

    return history, loss_history

def plot_result(history,loss_history):

    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)


    plt.figure(1)
    plt.plot(epochs, acc,'ro',label = 'Training accuracy')
    plt.plot(epochs, val_acc,'b',label = 'Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()

    plt.figure(2)
    plt.plot(epochs,loss,'bo',label = 'Training loss')
    plt.plot(epochs,val_loss,'b',label = 'Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.figure(3)
    plt.plot(epochs,loss_history.lr,'go',lable = 'Learning rate')
    plt.title('Learning rate')
    plt.legend()

    plt.show()

    with open('/home/dkkim/documents/tiny_log_2.txt','w') as output:
        output.write(str(history.history))
