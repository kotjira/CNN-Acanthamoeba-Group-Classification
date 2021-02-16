import tensorflow as tf
import PIL
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import plotly
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import plotly.graph_objs as go
from tensorflow import keras
from tensorflow.keras.models import Sequential
import pathlib

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    suppress_qt_warnings()


with open('history_model', 'rb') as file:
 his = p.load(file)
filepath='model1.h5'
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'
h1 = go.Scatter(y=his['loss'], 
 mode='lines', line=dict(
 width=2,
 color='blue'),
 name='loss'
 )
h2 = go.Scatter(y=his['val_loss'], 
 mode='lines', line=dict(
 width=2,
 color='red'),
 name='val_loss'
 )
 
data = [h1,h2]
layout1 = go.Layout(title='Loss',
 xaxis=dict(title='epochs'),
 yaxis=dict(title=''))
fig1 = go.Figure(data, layout=layout1)
#plotly.offline.iplot(fig1, filename='testMNIST')
predict_model = load_model(filepath) 
predict_model.summary()
with open(filepath_model, 'r') as f:
 loaded_model_json = f.read()
 predict_model = model_from_json(loaded_model_json)
 predict_model.load_weights(filepath_weights) 
 print('Loaded model from disk')

epochs = 10
acc = his['accuracy']
val_acc = his['val_accuracy']
loss=his['loss']
val_loss=his['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

