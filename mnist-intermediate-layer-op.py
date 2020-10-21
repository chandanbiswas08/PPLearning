from __future__ import print_function
import keras.backend as kb
import keras as k
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D

from mlxtend.data import loadlocal_mnist
import numpy as np
import tensorflow as tf
tf.random.set_seed(1236)
np.random.seed(1234)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])


batch_size = 128
num_classes = 10
epochs = 1


def load_mist_morphology(mnist_path, mnist_fractured_path):
    img_rows, img_cols = 28, 28

    x_train_orig, y_train_orig = loadlocal_mnist(
        images_path=mnist_path + '/train-images.idx3-ubyte',
        labels_path=mnist_path + '/train-labels.idx1-ubyte')

    x_train_frac, y_train_frac = loadlocal_mnist(
        images_path=mnist_fractured_path + '/train-images.idx3-ubyte',
        labels_path=mnist_fractured_path + '/train-labels.idx1-ubyte')

    x_test_orig, y_test_orig = loadlocal_mnist(
        images_path=mnist_path + '/t10k-images.idx3-ubyte',
        labels_path=mnist_path + '/t10k-labels.idx1-ubyte')

    x_test_frac, y_test_frac = loadlocal_mnist(
        images_path=mnist_fractured_path + '/t10k-images.idx3-ubyte',
        labels_path=mnist_fractured_path + '/t10k-labels.idx1-ubyte')

    # join fractures with original
    x_train = np.vstack((x_train_orig, x_train_frac))
    y_train = np.concatenate([y_train_orig, y_train_frac])
    x_test = np.vstack((x_test_orig, x_test_frac))
    y_test = np.concatenate([y_test_orig, y_test_frac])

    if kb.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_mist_morphology('data/mnist_plain', 'data/mnist_fractures')
#print ('{} {}'.format(x_train.shape, y_train.shape))
#print ('{} {}'.format(x_test.shape, y_test.shape))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = k.utils.to_categorical(y_train, num_classes)
y_test = k.utils.to_categorical(y_test, num_classes)

print(y_train.shape[0], 'train labels')
print(y_test.shape[0], 'test labels')


LOSS_OUTPUT=k.losses.categorical_crossentropy

input = Input(shape=(28, 28, 1), dtype='float32')
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
mid = Dense(128, activation='relu', name='fvec')(x)
y = Dense(num_classes, activation='sigmoid', name='yhat')(mid)
model = Model(inputs=input, outputs=[mid, y])

model.compile(loss=[None, LOSS_OUTPUT], loss_weights=[0, 1.0],
              optimizer=k.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
)
score = model.evaluate(x_test, y_test, verbose=0)
print('Evaluation results:', score)


def slant_cal(img):
    img_rows, img_cols = 28, 28
    img = np.asarray(img, dtype=float).squeeze(axis=-1)
    img = img * 255
    s12 = 0
    s22 = 0
    for i in range(img_rows):
        for j in range(img_cols):
            s12 = s12 + img[i][j] * (i-14) * (j-14)
            s22 = s22 + img[i][j] * (j-14) * (j-14)
    slant = np.arctan2(-s12, s22)
    if slant <= -0.3:
        return 0
    if slant > -0.3 and slant <= 0.3:
        return 1
    if slant > 0.3:
        return 2

def write_vecs(ovfp, pfp, sfp, s1fp, model, xlist, ylist):
    fvecs, yhat = model.predict_on_batch(xlist) # discard yhat
    for j in range(len(fvecs)):
        fvec = fvecs[j]
        dim = fvec.shape[0]
        vec_str = ''
        for i in range(0, dim):
            vec_str += '{:.6f}'.format(fvec[i]) + " "
        ovfp.write(vec_str + '\n')
        pfp.write("%d\n"%np.argmax(ylist[j]))
        if j < len(fvecs)/2:
            s1fp.write("0\n")
        else:
            s1fp.write("1\n")
        slant = slant_cal(xlist[j])
        sfp.write("%d\n"%slant)

ovfp = open('data/mnist_slant_frac_vecs.txt', "w")
pfp = open('data/mnist_slant_frac_p.txt', 'w')
sfp = open('data/mnist_slant_frac_s1.txt', 'w')
s1fp = open('data/mnist_slant_frac_s2.txt', 'w')
write_vecs(ovfp, pfp, sfp, s1fp, model, x_train, y_train)
write_vecs(ovfp, pfp, sfp, s1fp, model, x_test, y_test)
ovfp.close()
pfp.close()
sfp.close()
s1fp.close()