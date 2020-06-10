import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import  Dense,Conv2D, Flatten, Dropout, MaxPooling2D,concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import matplotlib.pyplot as plt
import datetime

tf.keras.backend.clear_session()

IMG_HEIGHT=150
IMG_WIDTH=150
input_image=keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
input_lidar=keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
input_radar=keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))

class JoinedGenerator(keras.utils.Sequence):
    def __init__(self, generator1, generator2,generator3):
        self.generator1 = generator1
        self.generator2 = generator2
        self.generator3 = generator3 

    def __len__(self):
        return len(self.generator1)

    def __getitem__(self, i):
        x1, y1 = self.generator1[i]
        x2, y2 = self.generator2[i]
        x3, y3 = self.generator3[i]
        return [x1, x2, x3], y1

    def on_epoch_end(self):
        self.generator1.on_epoch_end()
        self.generator2.on_epoch_end()
        self.generator3.on_epoch_end()

datagen_lidar = ImageDataGenerator()
# load and iterate training dataset
train_lidar = datagen_lidar.flow_from_directory('data_lidar/train/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)
# load and iterate validation dataset
val_lidar = datagen_lidar.flow_from_directory('data_lidar/validation/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)
# load and iterate test dataset
test_lidar = datagen_lidar.flow_from_directory('data_lidar/test/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)

datagen = ImageDataGenerator()
# load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('data/validation/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)
# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)

datagen_radar = ImageDataGenerator()
# load and iterate training dataset
train_radar = datagen_radar.flow_from_directory('data_radar/train/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)
# load and iterate validation dataset
val_radar = datagen_radar.flow_from_directory('data_radar/validation/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)
# load and iterate test dataset
test_radar = datagen_radar.flow_from_directory('data_radar/test/', target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary', batch_size=2)

x=Conv2D(16, 3, padding='same', activation='relu')(input_image)
output_image=MaxPooling2D()(x)

model_image = keras.Model(inputs=input_image,outputs=output_image, name='image_input')
model_image.summary()

y=Conv2D(16, 3, padding='same', activation='relu')(input_lidar)
output_lidar=MaxPooling2D()(y)

model_lidar = keras.Model(inputs=input_lidar,outputs=output_lidar, name='lidar_input')
model_lidar.summary()

w=Conv2D(16, 3, padding='same', activation='relu')(input_radar)
output_radar=MaxPooling2D()(w)

model_radar = keras.Model(inputs=input_radar,outputs=output_radar, name='radar_input')
model_radar.summary()

combined = concatenate([model_image.output, model_lidar.output, model_radar.output])

z=Conv2D(32, 3, padding='same', activation='relu')(combined)
z=MaxPooling2D()(z)
z=Conv2D(16, 3, padding='same', activation='relu')(z)
z=MaxPooling2D()(z)
z=Conv2D(64, 3, padding='same', activation='relu')(z)
z=MaxPooling2D()(z)
z=Conv2D(16, 3, padding='same', activation='relu')(z)
z=MaxPooling2D()(z)
z=Flatten()(z)
z=Dense(512, activation='relu')(z)
z=Dense(256, activation='relu')(z)
z=Dense(128, activation='relu')(z)
z = Dense(1, activation="linear")(z)
fusion = Model(inputs=[model_lidar.input, model_image.input,model_radar.input], outputs=z)
#keras.utils.plot_model(fusion, 'fusion.png', show_shapes=True)

fusion.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

checkpoint_path = "training_1/cp_fusion.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

training_generator = JoinedGenerator(train_it, train_lidar, train_radar)
validation_generator = JoinedGenerator(val_it, val_lidar,val_radar)
test_generator = JoinedGenerator(test_it, test_lidar,test_radar)
history = fusion.fit_generator(
    training_generator,
    steps_per_epoch=60,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=10,
    callbacks=[tensorboard_callback]
)	
loss= fusion.evaluate_generator(test_generator, steps=4)
fusion.save_weights('./checkpoints/my_checkpoint') 


#######################################TEST#################################################

image_mod=cv2.imread("1.jpg")
image_mod=cv2.resize(image_mod,(150,150))

image_mod_lidar=cv2.imread("3.jpg")
image_mod_lidar=cv2.resize(image_mod_lidar,(150,150))

image_mod_radar=cv2.imread("6.jpg")
image_mod_radar=cv2.resize(image_mod_lidar,(150,150))

val=fusion.predict([image_mod.reshape(-1,150,150,3),image_mod_lidar.reshape(-1,150,150,3),image_mod_radar.reshape(-1,150,150,3)])
print(val)
if(val[0][0]<0):
    print("car")
else:
    print("pedestrian")

image_mod=cv2.imread("2.jpg")
image_mod=cv2.resize(image_mod,(150,150))

image_mod_lidar=cv2.imread("4.jpg")
image_mod_lidar=cv2.resize(image_mod_lidar,(150,150))

image_mod_radar=cv2.imread("6.jpg")
image_mod_radar=cv2.resize(image_mod_lidar,(150,150))

val=fusion.predict([image_mod.reshape(-1,150,150,3),image_mod_lidar.reshape(-1,150,150,3),image_mod_radar.reshape(-1,150,150,3)])
print(val)
if(val[0][0]<0):
    print("car")
else:
    print("pedestrian")
