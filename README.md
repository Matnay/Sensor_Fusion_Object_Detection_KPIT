# Fusion of LiDAR, Depth Camera and Radar data for object classification

 - Ubuntu 16.04 Kinetic: ![](https://github.com/clynamen/nuscenes2bag/workflows/ubuntu_1604_kinetic/badge.svg)
<br>The output of multi_sensor_fusion based object detection based on the Faster_RCNN_inception_v2 architecture
<br>**Implemented in Tensorflow 1.14**

![Image EXAMPLE RESULT](https://github.com/Matnay/KPIT_Fusion_Object_Detection_DL/blob/master/Screenshot%20from%202020-06-22%2015-14-01.png)
![Image EXAMPLE RESULT](https://github.com/Matnay/KPIT_Deep_Learning/blob/master/Screenshot%20from%202020-06-04%2012-10-31.png)
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
