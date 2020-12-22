import os

import numpy as np
import tensorflow as tf


class MnistModel():

    def __init__(self, reset=False, v=False):
        if not os.path.exists('./model_data') or reset:
            self.__model = self.create_model(128, 128, v=v)
            self.__model.save('./model_data')
        else:
            self.__model = tf.keras.models.load_model('./model_data')

    @property
    def model(self):
        return self.__model

    @staticmethod
    def create_model(*sizes, v):
        layers = [tf.keras.layers.Flatten()]
        layers += [tf.keras.layers.Dense(size, activation='relu') for size in sizes]
        layers += [tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10, activation='softmax')]

        model = tf.keras.models.Sequential(layers=layers)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model.fit(x_train, y_train, epochs=5, verbose=int(v))
        print(model.evaluate(x_test, y_test, return_dict=True))

        return model

if __name__ == "__main__":
    mnistModel = MnistModel(v=True)
