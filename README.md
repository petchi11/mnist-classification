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
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Classification Report

              precision    recall  f1-score   support

           0       0.99      0.98      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.96      0.99      0.98      1032
           3       0.99      0.98      0.99      1010
           4       0.98      0.99      0.99       982
           5       0.98      0.98      0.98       892
           6       1.00      0.97      0.99       958
           7       0.97      0.99      0.98      1028
           8       0.97      0.98      0.97       974
           9       0.99      0.96      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

### Confusion Matrix

![[ 963    0    6    0    0    4    0    1    4    2]
 [   0 1131    3    0    0    0    1    0    0    0]
 [   0    2 1026    0    1    0    0    3    0    0]
 [   0    0    5  990    0    5    0    6    4    0]
 [   0    1    5    0  970    0    0    0    2    4]
 [   1    0    2    6    0  874    1    0    7    1]
 [   7    3    2    0    4    2  932    1    7    0]
 [   0    2   12    0    0    0    0 1013    1    0]
 [   3    0    8    1    0    0    0    4  954    4]
 [   1    5    0    1   10    3    0   12    6  971]]

### New Sample Data Prediction

!![download](https://user-images.githubusercontent.com/105230321/192107312-eeb33d71-4378-4ae1-ac5d-fde7f5692692.png)


## RESULT
