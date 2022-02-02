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
import random

class RadarPublisher():
	def __init__(self):
		self.image=np.zeros((600,600,3))
		self.image_pub=rospy.Publisher("radar_image",Image,queue_size=1)
		self.radar_sub_1=rospy.Subscriber("radar_front",RadarObjects,self.radarcallback_1)
		#self.radar_sub_2=rospy.Subscriber("radar_front_left",RadarObjects,self.radarcallback_2)
		#self.radar_sub_3=rospy.Subscriber("radar_front_right",RadarObjects,self.radarcallback_3)
		self.bridge=cv_bridge.CvBridge()
		self.id=0

	def radarcallback_1(self,msg):
		obj=RadarObject()
		for i in range(0,60):
			obj=msg.objects[i]
			self.image[600-int(2.5*(obj.pose.y+120))][int(2.5*(obj.pose.x+2))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.5)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.5)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.7)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.7)))]=255
		# cv2.imwrite("image_757"+str(self.id)+".jpg",self.image)
	        # self.id+=1
		# cv2.imshow("window",self.image)
		# cv2.waitKey(1)
		image_final=self.bridge.cv2_to_imgmsg(self.image,encoding="passthrough")
		self.image_pub.publish(image_final)
		self.image=np.zeros((600,600,3))

	def radarcallback_2(self,msg):
		obj=RadarObject()
		for i in range(0,60):
			obj=msg.objects[i]
			self.image[600-int(2.5*(obj.pose.y+120))][int(2.5*(obj.pose.x+2))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.5)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.5)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.7)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.7)))]=255

	def radarcallback_3(self,msg):
		obj=RadarObject()
		for i in range(0,30):
			obj=msg.objects[i]
			self.image[600-int(2.5*(obj.pose.y+120))][int(2.5*(obj.pose.x+2))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.3)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.3)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.5)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.5)))]=255
			self.image[600-int(2.5*(obj.pose.y+120+random.gauss(0,0.7)))][int(2.5*(obj.pose.x+2+random.gauss(0,0.7)))]=255
		

if __name__ == '__main__':
	rospy.init_node("radar_image")
	rate=rospy.Rate(10)
	rp=RadarPublisher()
	rospy.spin()
