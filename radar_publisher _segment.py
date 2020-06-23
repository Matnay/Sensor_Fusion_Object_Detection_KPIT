#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Float64
import cv2
import cv_bridge
from sensor_msgs.msg import Image
import numpy as np
from fusion.msg import RadarObjects,RadarObject
from matplotlib import pyplot as plt

class RadarPublisher():
	def __init__(self):
		self.image=np.zeros((600,600,3))
		self.image_pub=rospy.Publisher("radar_image",Image,queue_size=1)
		self.radar_sub=rospy.Subscriber("radar_front",RadarObjects,self.radarcallback)
		self.bridge=cv_bridge.CvBridge()
		self.id=0

	def radarcallback(self,msg):
		obj=RadarObject()
		for i in range(0,100):
			obj=msg.objects[i]
			#self.image[600-int(2.5*(obj.pose.y+120))][int(2.5*(obj.pose.x+2))]=255
		# cv2.imshow("window",self.image)
		# cv2.waitKey(1)
		self.image=cv2.imread("1.png")
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		pixel_values = self.image.reshape((-1, 3))
		# convert to float
		pixel_values = np.float32(pixel_values)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
		k = 3
		_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
		centers = np.uint8(centers)
		# flatten the labels array
		labels = labels.flatten()
		segmented_image = centers[labels.flatten()]
		segmented_image = segmented_image.reshape(self.image.shape)
		cv2.imshow("window",segmented_image)
		cv2.waitKey(1)
		# image_final=self.bridge.cv2_to_imgmsg(self.image,encoding="passthrough")
		# self.image_pub.publish(image_final)
		self.image=np.zeros((600,600,3))

if __name__ == '__main__':
	rospy.init_node("radar_image")
	rate=rospy.Rate(10)
	rp=RadarPublisher()
	rospy.spin()