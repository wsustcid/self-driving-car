<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-07-18 22:20:14
 * @LastEditTime: 2020-07-18 23:36:35
 * @Description:  
--> 

## Overview

### acc
This folder seems to contains a vehicle ACC (adaptive cruise control) demo code. (But now I am not sure how to use it...)

- *After running the `python setup.py test`, some packages will be installed and the acc package will be compiled, then a test code will run and finally a visualization window shows the pid control process.*


### annotations
This folder contains the download links of two annotated 2D object detection datasets. 
 - The annotated labels in Dataset 1 are car, truck and pedestrian and in Dataset 2 contains additional traffic lights annotations.


### challenges
In order to attract more people to contribute their code to this Open Source Self-Driving Car Project, Udacity break down the problem of making the car autonomous into three Udacity Challenges:
 - [Challenge #1: 3D Model for Camera Mount.](https://medium.com/udacity/challenge-1-3d-model-for-camera-mount-f5ffcc1655b5)
 - [Challenge #2: Using Deep Learning to Predict Steering Angles](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3)
 - [Challenge #3: Image-Based Localization](https://medium.com/udacity/challenge-3-image-based-localization-5d9cadcff9e7)

The file in this folder shows the leader board of the later two challenges, and the submission files are also included.


### datasets
This folder contains the download links of the two datasets (CH2 & CH3) used in the Challenge 2 and 3. Additionally, a new dataset CHX that contains VLP32 Lidar data is also provided.

A ROS Package named `udacity_launch` is provided to visualize the raw data (ros bag file) in rviz of the dataset.


### image-localization
This folder contains the code and documentation for community solutions to Udacity Challenge #3: Image-Based Localization. 


### images
The image folder contains the images used in the README file.


### sensor-info
This folder contains the camera calibration images and files.


### steering-models
This folder code and documentation for community solutions to Udacity Challenge #2: Predicting Steering Angles.


### vehicle-detection
This folder contains two community solutions of the vehicle detection problem.
  - darkflow
  - u-net