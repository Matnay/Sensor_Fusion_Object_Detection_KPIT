# Multi-Modal Sensor Fusion of LiDAR, Radar and Monocular Camera data for object detection

 - Ubuntu 16.04 Kinetic: ![](https://github.com/clynamen/nuscenes2bag/workflows/ubuntu_1604_kinetic/badge.svg)
<br>The output of multi_sensor_fusion based object detection based on the Faster_RCNN_inception_v2 architecture
<br>**Implemented in Tensorflow 1.14**
![Image EXAMPLE RESULT](https://github.com/Matnay/KPIT_Fusion_Object_Detection_DL/blob/master/Results/output.gif)
![Image EXAMPLE RESULT](https://github.com/Matnay/KPIT_Fusion_Object_Detection_DL/blob/master/Results/Screenshot%20from%202020-06-22%2015-14-01.png)
![Image EXAMPLE RESULT](https://github.com/Matnay/KPIT_Deep_Learning/blob/master/Results/Screenshot%20from%202020-06-04%2012-10-31.png)
# nuscenes to rosbag
Simple C++ tool for converting the [nuScenes](https://www.nuscenes.org/) dataset from [Aptiv](https://www.aptiv.com).

The tool loads the json metadata and then the sample files for each scene. The sample are converted in a suitable ROS msg and written to a bag. TF tree is also written.

Probably the original dataset is also collected by Aptiv using ROS, so most data has the same format.

![](images/ros_preview.png)

## Install
The tool is a normal ROS package. Place it under a workspace and build it with catkin.

## Usage

**Command-line arguments:**
`--dataroot`: The path to the directory that contains the 'maps', 'samples' and 'sweeps'.
`--version`: (optional) The sub-directory that contains the metadata .json files. Default = "v1.0-mini"


**Converting the 'mini' dataset:**

Convert one scene to a bag file, saved in a new directory:
Scene '0061' will be saved to 'nuscenes_bags/61.bag'
```
rosrun nuscenes2bag nuscenes2bag --scene_number 0061 --dataroot /path/to/nuscenes_mini_meta_v1.0/ --out nuscenes_bags/
```


Convert the entire dataset to bag files:
This processes 4 scenes simultaneously, however the scene numbers are not processed in numerical order.
```
rosrun nuscenes2bag nuscenes2bag --dataroot /path/to/nuscenes_mini_meta_v1.0/ --out nuscenes_bags/ --jobs 4
```
- [ ] Radar support

nuscenestobag
 - [clynamen](https://github.com/clynamen/)
 - [ChernoA](https://github.com/ChernoA)

LiDAR RADAR and monocular camera image fusion

Data trained on nuscenes dataset
After running rosbag run node to convert pointcloud into depth encoded 2d BEV frame representation 

```
rosrun fusion lidar_image_pub
```

Run multi_sensor_fusion.py to view classification results
```
rosrun fusion ros_multi_sensor_fusion.py
```
Object Detection implemented usinhg faster_rcnn_inception_v2
```
rosrun fusion faster_rcnn
```

#### Citing

If you use this work in an academic context, please cite the following publication:
```
@InProceedings{10.1007/978-981-16-7996-4_40,
author="Mathur, Pranay
and Kumar, Ravish
and Jain, Rahul",
editor="Chen, Joy Iong-Zong
and Wang, Haoxiang
and Du, Ke-Lin
and Suma, V.",
title="Multi-sensor Fusion-Based Object Detection Implemented on ROS",
booktitle="Machine Learning and Autonomous Systems",
year="2022",
publisher="Springer Singapore",
address="Singapore",
pages="551--563",
abstract="Mathur, PranayKumar, RavishJain, Rahul3D Perception of the environment in real-time is a critical aspect for object detection, obstacle avoidance, and classification in autonomous vehicles. This paper proposes a novel 3D object classifier that can exploit data from a LIDAR, a RADAR, and a monocular camera image after orthogonal projection. To achieve this, a learnable architecture is designed end-to-end, which fuses the detection results from multiple sensor modalities initially and exploits continuous convolution subsequently to achieve the desired levels of accuracy. An adaptive algorithm for prediction in real-time is used to automatically increase weightage to prediction results from a particular sensor modality which aids in keeping accuracy invariant to scene changes. To prevent the bias, we are using a training strategy which provides attention to the specific type of sensor. This strategy is inspired by dropout. The entire algorithm has been implemented on the Robot Operating System to make it easier to deploy and transfer. We have experimentally evaluated our method on the NuScenes dataset.",
isbn="978-981-16-7996-4"
}
```
#### Contact
- Pranay Mathur [![Gmail: Pranay Mathur](https://img.shields.io/badge/gmail-%23D14836.svg?&style=plastic&logo=gmail&logoColor=white)](mailto:matnay17@gmail.com) [![Linkedin: offjangir](https://img.shields.io/badge/-Pranay_Mathur-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/yash-jangir-6a71651a1)](https://www.linkedin.com/in/pranay-mathur1998/)
