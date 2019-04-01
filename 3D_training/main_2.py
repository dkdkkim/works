import model_3D as m3
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# set session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)



train_dir = '/home/dkkim/data/train'
val_dir = '/home/dkkim/data/valid'
log_data_dir = '/home/dkkim/data'
log_name = '3D_test_3'

model = m3.build_model()

train_x,train_y,val_x,val_y = m3.build_data(train_dir,val_dir)

history,loss_history = m3.training_model(model,train_x,train_y,val_x,val_y,log_data_dir,log_name,30,80,lr=0.01,gpus=2)

m3.plot_result(history,loss_history)