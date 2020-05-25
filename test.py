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
	self.id=0
	self.image=np.zeros((451,901,3))
	self.array=np.zeros((451,901))
	self.x=np.zeros((451,901))
	self.y=np.zeros((451,901))
	self.z=np.zeros((451,901))
	self.image_gs=np.zeros((600,1200))
	self.reconstruct=np.zeros((451,900))
	self.pub=rospy.Publisher("marker",MarkerArray,queue_size=1)
	self.bridge=cv_bridge.CvBridge()
	sub_1=rospy.Subscriber("lidar_top", PointCloud2, self.callback_kinect)
	#sub=rospy.Subscriber("/cam_front/raw",Image,self.image_cb)
	rate=rospy.Rate(10)
	rate.sleep()
      
  def image_cb(self,msg):
    self.image=self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
    self.half = cv2.resize(self.image, (0, 0), fx = 0.5, fy = 0.5)
    self.half=cv2.flip(self.half,0)
    self.x=self.half[:,:,2]
    self.y=self.half[:,:,0]
    self.z=self.half[:,:,1]
    

  def callback_kinect(self,data) :
    data_out = pc2.read_points(data, field_names=("x","y","z","intensity"), skip_nans=False)
    #print(data_out)
    i=0
    j=0
    self.id+=1
    for p in data_out:
      if(p[0]>-10 and p[0]<10 and p[1]>-10 and p[1]<10):
        # while(i<450):
        #   i+=1
        #   j=0
        #   while(j<900):
        #     j=j+1
            # self.image[:,:,0][i][j]=int(100*p[0])
            # self.image[:,:,1][i][j]=int(100*p[1])
            # self.image[:,:,2][i][j]=int(100*(p[2]+2))
    	self.image_gs[int(abs(30*(p[0]+10)))][int(abs(30*(p[1]+10)))]=255
    cv2.imwrite("ima"+str(self.id)+".jpg",self.image_gs)
    #self.image_gs=cv2.flip(self.image_gs,0)
    # cv2.imshow("image",self.image_gs)
    # cv2.waitKey(1)
    self.image_gs=np.zeros((600,600))

if __name__ == '__main__':
    try:
      rospy.init_node('listen', anonymous=True)
      fusion=Fusion()
      rospy.spin()
    except rospy.ROSInterruptException:
      print ("error")
      pass
