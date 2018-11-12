#!/usr/bin/env python
# coding: utf-8

# # TD : `keras` & perceptron multi-couches
# 
# Romain Tavenard
# Creative Commons CC BY-NC-SA

# Dans cette séance, nous nous focaliserons sur la création et l'étude de modèles
# de type perceptron multi-couches à l'aide de la librairie `keras`.
# 
# Pour cela, vous utiliserez la classe de modèles `Sequential()` de `keras`.
# Voici ci-dessous un exemple de définition d'un tel modèle :

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist, boston_housing
from keras.utils import to_categorical

import numpy as np

def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test

def prepare_boston():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    x_train = scaler_x.transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train[:,np.newaxis])
    y_train = scaler_y.transform(y_train[:,np.newaxis])
    y_test = scaler_y.transform(y_test[:,np.newaxis])
    return x_train, x_test, y_train, y_test

print("Boston dataset")
x_train, x_test, y_train, y_test = prepare_boston()
print(x_train.shape, y_train.shape)

print("MNIST dataset")
x_train, x_test, y_train, y_test = prepare_mnist()
print(x_train.shape, y_train.shape)

input_layer = Dense(units=128, 
                    activation="relu", 
                    input_dim=x_train.shape[1])

second_layer = Dense(units=128, 
                    activation="relu", 
                    input_dim=x_train.shape[1])


third_layer = Dense(units=y_train.shape[1], 
                    activation="softmax")

model = Sequential()
model.add(input_layer)
model.add(second_layer)
model.add(third_layer)

model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

model.fit(x_train, y_train, 
          epochs=10, batch_size=256, 
          verbose=2, validation_split=0.1)

y_pred = model.predict(x_train)

print(confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1)))

print(model.count_params())
