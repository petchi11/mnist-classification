# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries


### STEP 2:

### STEP 3:

Write your own steps

## PROGRAM
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
11501568/11490434 [==============================] - 0s 0us/step
X_train.shape
(60000, 28, 28)
X_test.shape
(10000, 28, 28)
single_image= X_train[100]
single_image.shape
(28, 28)
plt.imshow(single_image,cmap='gray')
<matplotlib.image.AxesImage at 0x7ff505794c90>

y_train.shape
(60000,)
X_train.min()
0
X_train.max()
255
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
0.0
X_train_scaled.max()
1.0
y_train[0]
5
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
numpy.ndarray
y_train_onehot.shape
(60000, 10)
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
<matplotlib.image.AxesImage at 0x7ff505291510>

y_train_onehot[500]
array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 5408)              0         
                                                                 
 dense (Dense)               (None, 32)                173088    
                                                                 
 dense_1 (Dense)             (None, 10)                330       
                                                                 
=================================================================
Total params: 173,738
Trainable params: 173,738
Non-trainable params: 0
_________________________________________________________________
# Choose the appropriate parameters
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
Epoch 1/5
938/938 [==============================] - 27s 28ms/step - loss: 0.2458 - accuracy: 0.9289 - val_loss: 0.0910 - val_accuracy: 0.9734
Epoch 2/5
938/938 [==============================] - 25s 27ms/step - loss: 0.0816 - accuracy: 0.9768 - val_loss: 0.0662 - val_accuracy: 0.9780
Epoch 3/5
938/938 [==============================] - 25s 27ms/step - loss: 0.0596 - accuracy: 0.9821 - val_loss: 0.0545 - val_accuracy: 0.9826
Epoch 4/5
938/938 [==============================] - 27s 28ms/step - loss: 0.0465 - accuracy: 0.9856 - val_loss: 0.0563 - val_accuracy: 0.9803
Epoch 5/5
938/938 [==============================] - 25s 27ms/step - loss: 0.0385 - accuracy: 0.9885 - val_loss: 0.0563 - val_accuracy: 0.9816
<keras.callbacks.History at 0x7ff5013cedd0>
metrics = pd.DataFrame(model.history.history)
metrics.head()
loss	accuracy	val_loss	val_accuracy
0	0.245837	0.928933	0.091014	0.9734
1	0.081567	0.976817	0.066225	0.9780
2	0.059638	0.982117	0.054469	0.9826
3	0.046493	0.985583	0.056280	0.9803
4	0.038486	0.988500	0.056321	0.9816
metrics[['accuracy','val_accuracy']].plot()
<matplotlib.axes._subplots.AxesSubplot at 0x7ff501362a90>

metrics[['loss','val_loss']].plot()
<matplotlib.axes._subplots.AxesSubplot at 0x7ff501299290>

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
[[ 972    0    1    0    1    1    1    1    1    2]
 [   0 1132    2    1    0    0    0    0    0    0]
 [   3    4 1012    1    2    0    0    6    2    2]
 [   0    0    4  992    0    8    0    3    2    1]
 [   0    0    0    0  975    0    0    0    0    7]
 [   2    0    1    4    0  882    2    0    1    0]
 [  10    3    1    0    6    4  932    0    2    0]
 [   0    4   10    2    2    0    0 1007    1    2]
 [   5    0    8    4    3    3    3    6  921   21]
 [   0    2    0    1    9    4    0    2    0  991]]
print(classification_report(y_test,x_test_predictions))
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.97      0.98      0.98      1032
           3       0.99      0.98      0.98      1010
           4       0.98      0.99      0.98       982
           5       0.98      0.99      0.98       892
           6       0.99      0.97      0.98       958
           7       0.98      0.98      0.98      1028
           8       0.99      0.95      0.97       974
           9       0.97      0.98      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

# This is formatted as code
Prediction for a single input

img = image.load_img('five.jpg')
type(img)
PIL.JpegImagePlugin.JpegImageFile
from tensorflow.keras.preprocessing import image
img = image.load_img('five.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
[5]
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
<matplotlib.image.AxesImage at 0x7ff5011e3ed0>

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
[5]
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here

### New Sample Data Prediction

Include your sample input and output for your hand written images.

## RESULT
