#!/usr/bin/env python

import os
import sys
import cv2
import cv_bridge
import numpy as np
import time
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    sys.exit(1)

# ROS related imports
import rospy
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from PIL import Image as img

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import cv2
from IPython.display import display
# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
MODEL_NAME = 'output_inference_graph_v2'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('annotations','label_map.pbtxt')
NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION

# Detection

class Detector:

	def __init__(self):
		self.object_pub = rospy.Publisher("objects", Detection2DArray, queue_size=1)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("image_lidar", Image, self.image_cb)
		self.sess = tf.Session(graph=detection_graph,config=config)
		#self.front_image_sub=rospy.Subscriber("cam_front/raw",Image,self.front_image_cb)
		#self.img_array=np.zeros((450,300))
		self.front_image_sub=rospy.Subscriber("debug_image",Image,self.front_image_cb)
		self.img_array=np.zeros((640,480))

	def front_image_cb(self,data):
		self.img_array=cv2.resize(self.bridge.imgmsg_to_cv2(data),(640,480))
		#self.img_array=cv2.resize(self.bridge.imgmsg_to_cv2(data),(450,300))
		time.sleep(0.2)
		#cv2.imshow("window",self.img_array)
		#cv2.waitKey(1)

	def drawBoundingBox(self,imgcv,array):
		cv2.rectangle(imgcv,(int(array[1]),250),(int(array[0]),320),(0,255,0),1)
		#cv2.rectangle(imgcv,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
		#cv2.putText(imgcv,label,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
		#cv2.imshow("window",imgcv)
		#cv2.waitKey(1) 

	def image_cb(self, data):
		objArray = Detection2DArray()
		try:
		    cv_image = self.bridge.imgmsg_to_cv2(data)
		except CvBridgeError as e:
		    print(e)
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image=cv_image[:,:,0]
		mod_image=np.zeros((600,600,3))
 		mod_image[:, :, 0]=image
 		mod_image[:, :, 1]=image
		mod_image[:, :, 2]=image  		
		image_np = mod_image.astype(np.uint8)
       	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
       	# Each box represents a part of the image where a particular object was detected.
		boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
		scores = detection_graph.get_tensor_by_name('detection_scores:0')
		classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		(boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
		objects,array=vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
		if(array!=None):
			array[0]=array[0]*640#450
			array[1]=array[1]*640#450
			array[2]=array[2]*480#300
			array[3]=array[3]*480#300
			print(array)
			self.drawBoundingBox(self.img_array,array)
		#display(img.fromarray(image_np))
		#cv2.imshow("window",image_np))
		h1, w1 = image_np.shape[:2]
		h2, w2 = self.img_array.shape[:2]
		#create empty matrix
		vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
		#combine 2 images
		vis[:h1,:w1,:3] = image_np
		vis[:h2,w1:w1+w2,:3] =self.img_array
		cv2.imshow("window",vis)
		cv2.waitKey(1)
			
if __name__=='__main__':
	rospy.init_node('detector_node')
	obj=Detector()
	try:
	    rospy.spin()
	except KeyboardInterrupt:
	    print("ShutDown")
	cv2.destroyAllWindows()
