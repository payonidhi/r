import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SentimentAnalysisModel(keras.Model):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
