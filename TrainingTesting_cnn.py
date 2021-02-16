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

dataset = "D:\project\CNN\CNN_test1\Dataset"
data_dir = pathlib.Path(dataset)

image_count = len(list(data_dir.glob('*/*.jpg')))
print('image_count_fruit :',image_count)

G1 = list(data_dir.glob('G1/*'))
PIL.Image.open(str(G1[0]))

batch_size = 30
img_height = 100
img_width = 100

train = tf.keras.preprocessing.image_dataset_from_directory(
 data_dir,
 validation_split=0.2, 
 subset='training',
 seed=123,
 image_size=(img_height, img_width),
 batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory( data_dir,validation_split=0.2,subset='validation',seed=120,image_size=(img_height, img_width),batch_size=batch_size)

class_names = train.class_names
print(class_names)

plt.figure(figsize=(8, 8))
for images, labels in train.take(1):
 for i in range(12):
  ax = plt.subplot(3, 4, i + 1)
  plt.imshow(images[i].numpy().astype('uint8'))
  plt.title(class_names[labels[i]])
  plt.axis('off')

for image_batch, labels_batch in train:
 print(image_batch.shape)
 print(labels_batch.shape)
 break

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255) 
normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = 4
epochs=15
model = Sequential([
 layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(num_classes)
])

model.compile(optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy']) 
model.summary()

his = model.fit(
 train,
 validation_data=val_ds,
 epochs=epochs
)

with open('history_model', 'wb') as file:
 p.dump(his.history, file)
 
filepath='model1.h5'
model.save(filepath)
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'
model_json = model.to_json()
with open(filepath_model, 'w') as json_file:
 json_file.write(model_json)
 
 model.save_weights('weights_model.h5')
 print('Saved model to disk')

plt.show()