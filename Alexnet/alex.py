import tensorflow as tf
import alex_model as am
import keras
import os

from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))



# from tensorflow.python.client import device_lib
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

from keras.utils.training_utils import multi_gpu_model


model = am.build_model()

model = multi_gpu_model(model, gpus=2)


# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

train_data, val_data = am.build_data('/home/dkkim/downloads/nodule_image/train/','/home/dkkim/downloads/nodule_image/validation/')

history = model.fit_generator(train_data, steps_per_epoch=2000, epochs=50, validation_data=val_data, validation_steps=800)

# model_2.save(model_weight)

log_data_dir = '/home/dkkim/works/Alexnet/'

model_weight=os.path.join(log_data_dir,'tiny_2_imagenet.h5')
model_arch=os.path.join(log_data_dir,'tiny_2_imagenet.json')

model.save(model_weight)
with open(model_arch,'w') as f:
    f.write(model.to_json())