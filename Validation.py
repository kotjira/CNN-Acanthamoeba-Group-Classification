import requests
from tensorflow import keras
from IPython.display import Image
from io import BytesIO
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

batch_size = 30
img_height = 100
img_width = 100

#-------load_model-------------
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
#-------load_model-------------

test_path = ('Dataset/Test/176.JPG')
img = keras.preprocessing.image.load_img(
 test_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = predict_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("Group1",score[0])
print("Group2",score[1])
print("Group3",score[2])

img=mpimg.imread(test_path)
plt.imshow(img)

if score[0]==np.max(score) :
 fruit = "Group1"
elif score[1]==np.max(score) :
 fruit = "Group2"
elif score[2]==np.max(score) :
 fruit = "Group3"
print(
 "ภาพนี้คือ {} {:.2f}%."
 .format(fruit, 100 * np.max(score))
)

plt.show()