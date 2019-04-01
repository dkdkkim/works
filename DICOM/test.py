from glob import glob
import numpy as np
from sklearn.utils import shuffle

train_dir = '/home/dkkim/data/train'
val_dir = '/home/dkkim/data/valid'

## train

train_path_0 = glob(train_dir + '/class0/*.npy')
train_path_1 = glob(train_dir + '/class1/*.npy')

train_path_0 = train_path_0[0:10]
train_path_1 = train_path_1[0:10]

train_X = np.empty((len(train_path_0) + len(train_path_1), 48, 48, 48))
train_Y = np.zeros((len(train_path_0) + len(train_path_1), 2))

for pathIdx, path in enumerate(train_path_0):
    train_X[pathIdx] = np.load(path)
    train_Y[pathIdx, 0] = 1

for pathIdx, path in enumerate(train_path_1):
    train_X[pathIdx + len(train_path_0)] = np.load(path)
    train_Y[pathIdx + len(train_path_0), 1] = 1

train_X = np.expand_dims(train_X, axis=4)

tX, tY = train_X, train_Y
# tX, tY = shuffle(train_X, train_Y, random_state=0)

print tX.shape, tY.shape

x = np.expand_dims(np.load(path), axis=3)

print tX[19].shape, x.shape

if np.all([tX[19],x]): print 'ok'

for pathIdx, path in enumerate(train_path_0):
    if np.all([tX[pathIdx],np.expand_dims(np.load(path),axis=3)]):
        print pathIdx, 'ok'