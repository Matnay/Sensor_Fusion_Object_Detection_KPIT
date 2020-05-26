import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import matplotlib.pyplot as plt
import rospy,cv_bridge,time
from sensor_msgs.msg import Image
import numpy as np
tf.compat.v1.enable_eager_execution()

class Prediction:
	def __init__(self):
		self.bridge=cv_bridge.CvBridge()
		self.IMG_HEIGHT = 200
		self.IMG_WIDTH = 200
		#self.sub=rospy.Subscriber("/cam_front/raw",Image,self.image_cb)
		self.image=cv2.imread("5.jpg")
		self._session=tf.Session()
		self.model = Sequential([
    		Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH ,3)),
    		MaxPooling2D(),
    		Conv2D(32, 3, padding='same', activation='relu'),
    		MaxPooling2D(),
    		Conv2D(64, 3, padding='same', activation='relu'),
    		MaxPooling2D(),
   			Flatten(),
    		Dense(512, activation='relu'),
    		Dense(1)])
		self.model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
		# Restore the weights
		self.model.load_weights('./checkpoints/lidar_checkpoint')
		self.model._make_predict_function()
		self.model.summary()	
		val=self.model.predict([self.image.reshape(-1,200,200,3)])
		if(val[0][0]>0):
		    print("person")
		else:
		    print("car")

if __name__ == '__main__':
	prediction=Prediction()
