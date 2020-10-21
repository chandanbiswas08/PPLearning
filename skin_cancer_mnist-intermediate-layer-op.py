from __future__ import print_function
import keras.backend as kb
import keras as k
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import math
import numpy as np
import tensorflow as tf
tf.random.set_seed(1236)
np.random.seed(1234)

batch_size = 128
num_classes = 7
epochs = 10


def load_hmist(hmnist_path):
    img_rows, img_cols = 28, 28
    TRAIN_TEST_DIV = 0.85
    df = pd.read_csv(hmnist_path)
    df = df.sample(frac=1, random_state=1)
    DATABASE_SIZE = len(df)
    SPLIT_INDEX = math.ceil(DATABASE_SIZE * TRAIN_TEST_DIV)

    df['sex'].loc[df['sex'] == 'male'] = 1
    df['sex'].loc[df['sex'] == 'female'] = 0

    df['age'].loc[df['age'] <= 30] = 0
    df.loc[(df['age'] > 30) & (df['age'] <= 60), 'age'] = 1
    df['age'].loc[df['age'] > 60] = 2


    dataset = np.asarray(df.iloc[:,:-3].values, dtype="float32")
    # np.savetxt('hmnist_784', dataset/255, fmt='%1.6f')
    target = np.asarray(df.iloc[:,-3].values, dtype="int")
    age = np.asarray(df.iloc[:,-2].values, dtype="int")
    sex = np.asarray(df.iloc[:, -1].values, dtype="int")

    x_train = dataset[0:SPLIT_INDEX]
    y_train = target[0:SPLIT_INDEX]
    age_train = age[0:SPLIT_INDEX]
    sex_train = sex[0:SPLIT_INDEX]
    x_test = dataset[SPLIT_INDEX:DATABASE_SIZE]
    y_test = target[SPLIT_INDEX:DATABASE_SIZE]
    age_test = age[SPLIT_INDEX:DATABASE_SIZE]
    sex_test = sex[SPLIT_INDEX:DATABASE_SIZE]

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    return x_train, y_train, x_test, y_test, age_train, age_test, sex_train, sex_test



x_train, y_train, x_test, y_test, age_train, age_test, sex_train, sex_test = load_hmist('data/skin_cancer_mnist.csv')
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
mid = Dense(256, activation='relu', name='fvec')(x)
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

def write_vecs(ovfp, pfp, sfp, s1fp, model, xlist, ylist, age, sex):
    fvecs, yhat = model.predict_on_batch(xlist) # discard yhat
    for j in range(len(fvecs)):
        fvec = fvecs[j]
        dim = fvec.shape[0]
        vec_str = ''
        for i in range(0, dim):
            vec_str += '{:.6f}'.format(fvec[i]) + " "
        ovfp.write(vec_str + '\n')
        pfp.write("%d\n"%np.argmax(ylist[j]))
        sfp.write("%d\n"%sex[j])
        s1fp.write("%d\n"%age[j])


ovfp = open('data/skin_cancer_mnist_256_dense', "w")
pfp = open('data/skin_cancer_mnist_dx.txt', 'w')
sfp = open('data/skin_cancer_mnist_sex.txt', 'w')
s1fp = open('data/skin_cancer_mnist_age.txt', 'w')
write_vecs(ovfp, pfp, sfp, s1fp, model, x_train, y_train, age_train, sex_train)
write_vecs(ovfp, pfp, sfp, s1fp, model, x_test, y_test, age_test, sex_test)
ovfp.close()
pfp.close()
sfp.close()
s1fp.close()





