import rospy
import tensorflow as tf 
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import  Dense,Conv2D, Flatten, Dropout, MaxPooling2D,concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import rospy,cv_bridge,time
from sensor_msgs.msg import Image
import numpy as np

tf.compat.v1.enable_eager_execution()
class Fusion():
 	def __init__(self):
 		self.IMG_HEIGHT = 150
		self.IMG_WIDTH = 150
		self.night=False
		self.image=np.zeros((900,1600,3))
		self.image_lidar=np.zeros((600,600,3))
		self.sub=rospy.Subscriber("/cam_front/raw",Image,self.image_cb)
		self.lidar_sub=rospy.Subscriber("image_lidar",Image,self.lidar_image_cb)
		self._session=tf.Session()
		self.bridge=cv_bridge.CvBridge()
 		
 		self.input_image=keras.Input(shape=(self.IMG_HEIGHT,self.IMG_WIDTH,3))
		self.input_lidar=keras.Input(shape=(self.IMG_HEIGHT,self.IMG_WIDTH,3))

		self.x=Conv2D(16, 3, padding='same', activation='relu')(self.input_image)
		self.output_image=MaxPooling2D()(self.x)

		self.model_image = keras.Model(inputs=self.input_image,outputs=self.output_image, name='image_input')
		self.model_image.summary()

		self.y=Conv2D(16, 3, padding='same', activation='relu')(self.input_lidar)
		self.output_lidar=MaxPooling2D()(self.y)

		self.model_lidar = keras.Model(inputs=self.input_lidar,outputs=self.output_lidar, name='lidar_input')
		self.model_lidar.summary()

		self.combined = concatenate([self.model_image.output, self.model_lidar.output])

		self.z=Conv2D(32, 3, padding='same', activation='relu')(self.combined)
		self.z=MaxPooling2D()(self.z)
		self.z=Conv2D(64, 3, padding='same', activation='relu')(self.z)
		self.z=MaxPooling2D()(self.z)
		self.z=Conv2D(16, 3, padding='same', activation='relu')(self.z)
		self.z=MaxPooling2D()(self.z)
		self.z=Flatten()(self.z)
		self.z=Dense(512, activation='relu')(self.z)
		self.z=Dense(128, activation='relu')(self.z)
		self.z = Dense(1, activation="linear")(self.z)
		self.fusion = Model(inputs=[self.model_lidar.input, self.model_image.input], outputs=self.z)
		#keras.utils.plot_model(fusion, 'fusion.png', show_shapes=True)
		self.fusion.load_weights('./checkpoints/my_checkpoint')
		self.fusion.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
		self.fusion.load_weights('./checkpoints/my_checkpoint')
		self.fusion._make_predict_function()


	def create_model(self):
		model = Sequential([
    	Conv2D(16, 3, padding='same', activation='relu', input_shape=(150,150,3)),
    	MaxPooling2D(),
	    Conv2D(32, 3, padding='same', activation='relu'),
    	MaxPooling2D(),
    	Conv2D(64, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Flatten(),
    	Dense(512, activation='relu'),
    	Dense(1)])
		model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
		return model


	def image_cb(self,msg):
		self.image=self.bridge.imgmsg_to_cv2(msg)
		# if(False):
		# 	self.night=True

	def lidar_image_cb(self,msg):
		self.image_lidar=np.zeros((600,600,3))
		self.image_lidar=self.bridge.imgmsg_to_cv2(msg)
		self.prediction()
		# if(self.night==False):
		# 	self.prediction()
		# else: 
		# 	model=self.create_model()
		# 	model.load_weights('./checkpoints/lidar_checkpoint')
		# 	image_mod_lidar=cv2.resize(self.image_lidar,(150,150))
		# 	val=model.predict(image_mod_lidar.reshape(-1,150,150,3))
		# 	if(val[0][0]>0):
		# 		print("person")
		# 	else:
		# 		print("car")


	def prediction(self):
		image_mod=cv2.resize(self.image,(150,150))
		image_mod_lidar=cv2.resize(self.image_lidar,(150,150))
		val=self.fusion.predict([image_mod.reshape(-1,150,150,3),image_mod_lidar.reshape(-1,150,150,3)])
		if(val[0][0]>0):
		    print("person")
		else:
		    print("car")

if __name__ == '__main__':
	rospy.init_node("fusion_DL")
	rate=rospy.Rate(10)
	fus=Fusion()
	rospy.spin()
	
