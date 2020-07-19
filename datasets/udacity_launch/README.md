<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-05-18 09:49:52
 * @LastEditTime: 2020-07-19 16:57:13
--> 
## Overview
This package contains the tools for visualizing and extrcting point clouds, images and control messages from the ros bag file. Finally, a new driving behavior dataset can be created using these information.
<img src=./assets/lidar-32.gif />

## Download Dataset
Download compressed file such as CHX_001.tar.gz and CH03_002.bag.tar.gz. Then decompress them
```
tar -zxvf FILE_NAME.tar.zip -C /FOLDER_NAME 
```

## Installation
```python
# To Utilize compressed image topics
sudo apt-get install ros-kinetic-image-transport*
```
## Usage

### Play back data
Steps:
1. copy the udacity_launch package to you catkin workspace;
2. compile and source so that it is reachable.
3. play and visulize data

```python
# Terminal 1
source ~/catkin_ws/devel/setup.bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roscore

# Terminal 2: paly the raw data
cd ~/udacity-driving-data/CHX_001
rosbag play --clock *.bag

# Terminal 3: covert raw data (\velodyne_packets) to proper topics (velodyne_points))
# to visualize it in rviz
# bag_play.launch is used for CH2
roslaunch udacity_launch bag_play.launch velodyne_packets2pointcloud:=True
# For CH3 and CHX, we should use bag_play_new.launch

# Terminal 4: visualize images and point clouds
roslaunch udacity_launch rviz.launch
```

### Extrating images and msgs
We read and dump data from the rosbag using the tools provided by <https://github.com/rwightman/udacity-driving-reader>, and save them to the `CHX_001_output` folder.

```python
# Dump to images + CSV
cd udacity-driving-reader/
./run-bagdump.sh -i [absolute dir with folders containing bag files] -o [absolute output dir] -- [args to pass to python script]
```
```python
## After dumping,we have 
CHX_001_output/
├── brake.csv
├── cal_loop.yaml
├── camera.csv
├── /center
├── /left
├── /right
├── gear.csv
├── gps.csv
├── imu.csv
├── interpolated.csv
├── steering.csv
└── throttle.csv
```

### Save PointCloud2 msg to pcd file.
See <http://wiki.ros.org/pcl_ros#pointcloud_to_pcd> for more useful tools.
```python
# install pcl_ros (under the data folder)
sudo apt-get install ros-kinetic-pcl-ros

# play bag
rosbag play --clock *.bag

# covert velodyne raw packets msg to standard PointCloud2 message
roslaunch udacity_launch bag_play.launch velodyne_packets2pointcloud:=True

# covert point cloud to pcd (under the to be saved folder)
rosrun pcl_ros pointcloud_to_pcd:=/velodyne_points

```

### Creat a new time synchronized dataset
This dataset is used for driving policy learning.

- 

### Others

**Other tool which can be referred:**
- https://github.com/shanealynn/ros_csv_extraction/


**log and decompress:**
```python
# To log data in separate bagfiles and compressed image
roslaunch udacity_launch logging.launch bagPath:="/media/Data/UdacitySDC/udacity-datasetN"

# To convert existing log to seperate bagfiles and compressed image
rosbag  play --clock old_single_huge_bagfile.bag
roslaunch udacity_launch logging.launch republish_raw2compressed_images:=true bagPath:="/media/Data/UdacitySDC/udacity-datasetN"


# Compressed bagfiles
# Each of the bagfiles are already compressed, so there is no benefit
# of compressing the folders to compressed file other than having a single file for the whole folder.

# The bagfiles decompress on the fly and may cause slower performance and you may want to decompress them before you run, size would increase by around 10% only.

rosbag decompress *.bag

```

## CHANGE LOG
1. update the velodyne calibration file in bag_play.launch to show point cloud data more accurate.
2. update display.rviz to show point cloud in rviz.
3. extract images and msgs into folder and csv file.
4. save raw point cloud data in .bag file into .pcd file and velodyne point info. 
5. finally registering point data with images and correct steering angles of left and right image. 
