#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from roslib import message
from geometry_msgs.msg import PoseStamped 
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from visualization_msgs.msg import MarkerArray,Marker
import visualization_msgs
import time

class Fusion:
	def __init__(self):
		self.image=np.zeros((451,901))
		self.array=np.zeros((451,901))
		self.red=np.zeros((451,901))
		self.green=np.zeros((451,901))
		self.blue=np.zeros((451,901))
		self.reconstruct=np.zeros((451,900))
		self.pub=rospy.Publisher("marker",MarkerArray,queue_size=1)
		self.bridge=cv_bridge.CvBridge()
		sub_1=rospy.Subscriber("lidar_top", PointCloud2, self.callback_kinect)
		sub=rospy.Subscriber("/cam_front/raw",Image,self.image_cb)
		rate=rospy.Rate(10)
		rate.sleep()
	    
	def image_cb(self,msg):
		self.image=self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
		self.half = cv2.resize(self.image, (0, 0), fx = 0.5, fy = 0.5)
		self.half=cv2.flip(self.half,0)
		self.red=self.half[:,:,2]
		self.blue=self.half[:,:,0]
		self.green=self.half[:,:,1]
		

	def callback_kinect(self,data) :
		data_out = pc2.read_points(data, field_names=("x","y","z","intensity"), skip_nans=False)
		#print(data_out)
		marker_array=MarkerArray()
		id=0	
		for p in data_out:
			if(p[0]>-5 and p[0]<5 and p[1]>0.1):
				marker=Marker()
				marker.header.frame_id="base_link"	
				marker.type=visualization_msgs.msg.Marker.CUBE
				marker.id=id
				marker.header.stamp=rospy.Time.now()
				marker.scale.x=0.1
				marker.scale.y=0.1
				marker.scale.z=0.1
				marker.action=visualization_msgs.msg.Marker.ADD
				marker.color.a=1.0
				red=self.red[int(30*(p[2]+2.3))][int(45*(p[0]+5))]
				blue=self.blue[int(30*(p[2]+2.3))][int(45*(p[0]+5))]
				green=self.green[int(30*(p[2]+2.3))][int(45*(p[0]+5))]
				#print(str(int(40*(p[0]+5)))+" "+str(int(70*(p[2]+2.3))))
				#self.reconstruct[450-int(90*(p[2]+1.9))][int(100*(p[0]+5))]=255
				marker.color.r=float(red)/255
				marker.color.g=float(green)/255
				marker.color.b=float(blue)/255
				marker.pose.position.x=p[0]
				marker.pose.position.y=p[1]
				marker.pose.position.z=p[2]+1.9
				if(id > 10000):
					id=0
					marker_array=MarkerArray()
					pass
				marker_array.markers.append(marker)
				id+=1
		self.pub.publish(marker_array)


if __name__ == '__main__':
    try:
    	rospy.init_node('listen', anonymous=True)
    	fusion=Fusion()
    	rospy.spin()
    except rospy.ROSInterruptException:
    	print ("error")
        pass
