<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-05-17 16:49:25
 * @LastEditTime: 2020-07-19 16:09:30
--> 
<img src="../images/car.jpeg" alt="Self-Driving Car" width="800px">

# Udacity Self Driving car Challenges
News:
1. https://medium.com/udacity/were-building-an-open-source-self-driving-car-ac3e973cd163
2. https://medium.com/udacity/challenge-1-3d-model-for-camera-mount-f5ffcc1655b5
3. https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3
4. https://medium.com/udacity/challenge-3-image-based-localization-5d9cadcff9e7
5. https://medium.com/udacity/open-sourcing-3%C2%BD-hours-of-driving-data-with-lidar-b9258496a78b




## Platform
__The car__: A 2016 Lincoln MKZ;
**Sensors and parts**: 2 Velodyne VLP-16 LiDARs, 1 Delphi radar, 3 Point Grey Blackfly cameras, an Xsens IMU, an ECU, a power distribution system, and more! 
**Operating System**: ROS


## Challenge #1: 3D Model for Camera Mount
Our Point Grey Blackfly cameras, while amazing, only provided simple tripod mounts, which don’t provide support for the lens when mounted in a car. Cars are often bumpy, unpredictable, and the data we record must be consistent, otherwise hours of driving are rendered useless. We also noticed (via YouTube videos) that both Nvidia and Comma use 3D printed hardware for mounting their cameras, which inspired the first beta challenge: Design a mount for our cameras to attach to a GoPro mount! Deliverable: a 3d model.

You can read more information about [the results of this challenge](https://medium.com/udacity/challenge-1-3d-model-for-camera-mount-f5ffcc1655b5), and find [the mount open sourced on GitHub](https://github.com/udacity/camera-mount).

## Challenge #2: Using Deep Learning to Predict Steering Angles
You may have seen this [incredible video from Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), one of our Nanodegree program partners, which highlights their efforts at teaching a car how to drive using only cameras and deep learning. Their DAVE-2 deep learning system is capable of driving in many different weather conditions, avoiding obstacles, and even going off-road! You may have noticed their setup looks pretty similar to our 2016 Lincoln MKZ, and that’s for good reason. One of the first ways that we want to get this car on the road is to implement a similar end-to-end solution, and release that to the world for free. 

The second challenge for the Udacity Self-Driving Car initiative is to replicate these results using a convolutional neural network that you design and build!

End-to-end solutions like this, where a single network takes raw input (camera imagery) and produces a direct steering command. By letting the car figure out how to interpret images on its own, we can skip a lot of the complexity that exists in manually selecting features to detect, and drastically reduce the cost required to get an autonomous vehicle on the road by avoiding LiDAR-based solutions.

### Challenge Overview
The purpose of the challenge is to take image frames from a camera mounted to the windshield of our car, and predict the appropriate steering angle using convolutional neural networks and deep learning. Training can take as long as you’d like, but the final network itself has to run in **real-time** (process a frame faster than 1/20th of a second).

While the dataset may include auxilary information, you may **only use camera imagery and steering wheel angle** information to train.

The metric used to determine the winner will be measured and evaluated by the network’s performance in simulation. Teams will be provided with two ROSbag-based datasets; one will be accompanied by steering angles, but the other won’t. Training should be completed on the first dataset, and testing on the second. Teams will be required to generate a CSV file indicating their networks steering decisions, and this file will be uploaded and processed to determine team rankings.

**Evaluation Metric**
Root Mean Square Error. 

### Get Started
1. Download the [datasets here](https://github.com/udacity/self-driving-car).
2. Install Ubuntu 14.04 in a [virtual machine](https://www.virtualbox.org/) or directly onto your system.
3. Install [ROS](http://wiki.ros.org/) to playback data and convert into different formats.
4. Begin building and testing your convolutional neural network using TensorFlow.

### Data Format
All sensor data including imagery and steering wheel angles is provided in the ROSbag format. To playback this data, you will need to install ROS on a Ubuntu Linux platform and test from there. Additionally, you can convert the data into any format you like.

### Real-Time Performance
Essentially, your network has to process 15+ frames a second.



## Challenge #3: Image-Based Localization
Challenge #3 will deal with one of the most widely studied aspects of robotics engineering: localization. To operate safely, a self-driving vehicle must literally know where it is in the world, and this is not possible simply by relying on GPS, where accuracy can vary wildly depending on conditions. Unlike many other localization solutions, **we are not going to rely on LIDAR, but camera imagery.**

### Challenge Overview
Images hold a lot of information, and can actually contain more landmarks in their field of view than an equivalent LIDAR frame. By processing imagery in real-time and comparing those images to previous drives in the same area, you can actually get a localization solution that is good enough for use in navigation. Think of it this way: When you are walking down a street that you’ve traversed several times before, you know where you are because of how close you are to a certain building, intersection, or bridge. This information is all visual, and we can teach computers how to make the same decisions based off of landmarks that they can interpret.

This challenge will be heavy in image processing and tools like OpenCV. You will need to build a pipeline that can take a frame from our dashboard camera, process it, and compare it to a database of previous drives. This database doesn’t need to be a directory of images, and you’ll actually find that it will be too slow to index regular imagery. *If you don’t have any experience with things like fourier transforms or converting images to grayscale, you may want to join a team with expertise in these areas.*


### Rules
- You must produce a localization solution (latitude, longitude in the same format as the dataset) using only imagery from the front-facing center camera.
- You can train using the GPS localization solution recorded in the ROS bags in the datasets.
- Udacity will provide the teams with two datasets, training and testing. The training set will be accompanied by GPS location values for each frame, but the testing/evaluation set will not. The teams will then build a model on the training data, use it to predict on the testing data, and create a file with predicted localization solutions for the test set (again for each frame).

### Evaluation Metric
Root Mean Square Error. 

### Get Started
1. Download the [datasets here](https://github.com/udacity/self-driving-car)
2. Install Ubuntu 14.04 in a virtual machine or directly onto your system.
3. Install ROS to playback data and convert into different formats


## Open Sourcing 3½ Hours of Driving Data (With LIDAR!)
To date, we have open sourced over 4 hours and 200GB of driving data to assist the participants of Challenge #2 and Challenge #3.

### Overview
The path this data covers is going to start to become a familiar one; from the Udacity office in Mountain View to San Francisco along El Camino Real. All of the required steering and GPS data is present for Challenge #2 and Challenge #3, and the distance traversed covers where the test data for Challenge #3 will be pulled from, albeit on a different day.

This release is in the same format as the last one (so it’s a 30GB download instead of 200GB), with one cool extra: **LIDAR! **

Along with an image frame from our cameras, we also include latitude, longitude, gear, brake, throttle, steering angles and speed.

