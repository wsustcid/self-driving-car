#!/usr/bin/env python

'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-18 11:15:26
@LastEditTime: 2020-05-26 11:36:42
'''

""" 
TODO:
1. covert raw PointCloud2 data into x,y,z
2. save to file


reference:
http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html
https://blog.csdn.net/Fourier_Legend/article/details/83656798
https://answers.ros.org/question/202787/using-pointcloud2-data-getting-xy-points-in-python/

"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

import csv

class cloud2file():
    def __init__(self):
        # Init node
        rospy.init_node('cloud2file')

        rospy.on_shutdown(self.shutdown)
        
        dataset_path = rospy.get_param("~dataset_path", 
                            "/media/ubuntu16/Documents/Datasets/Udacity/test/")
        csv_name     = rospy.get_param("~csv_name", "test.csv")

        self.filename  = dataset_path + csv_name

        # cloud subscriber
        self.cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2,
                                          self.cloud_callback, queue_size=1)
        


    ## Read sonar data
    def cloud_callback(self, msg):
        
        self.header = msg.header
        self.height = msg.height
        self.width = msg.width
        self.fields = msg.fields
        self.point_step = msg.point_step
        self.row_step = msg.row_step
        self.data = msg.data
        self.is_dense = msg.is_dense
        
        print('==================')
        print("seq:{}".format(self.header.seq))
        print("Stamp:{}".format(self.header.stamp.secs*1e9+self.header.stamp.nsecs))
        print("height:{}".format(self.height)) # 1
        print("width:{}".format(self.width)) # 50000~60000
        print(self.fields)
        print(self.point_step) # 22
        print(self.row_step) # 0
        #print(self.data)
        print(self.is_dense) # True

        gen = point_cloud2.read_points(msg)
        
        #self.sonar_stamp = rospy.get_time()
        
        # compute ecuild distance and save them to a list
        #self.dist = [sqrt((msg.points[i].x)**2+(msg.points[i].y)**2) for i in range(16)]
    

    def data_collection(self, event):

        
        if (self.enable_data and self.sonar_received and self.cmd_received):
            
            self.sonar_received = False
            self.cmd_received   = False

            row = self.dist + self.cmd

            # save 1 time sequence data for each sample row
            with open(self.filename_1, 'a') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                self.count += 1
                print("save data: %d" % self.count)
            
            # save 16 time sequence data for each sample row 
            if self.Q16.qsize() == 15:
                self.Q16.put(row)
                data = [self.Q16.get() for i in range(16)] # size: (16,18)
                
                # reshape
                row16 = []
                for item in data: 
                    row16 += item # size: (16x18,)
                with open(self.filename_16, 'a') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row16)

                # drop first data
                # put back to the queue
                for item in data[1:]:
                    self.Q16.put(item)
            else: 
                self.Q16.put(row)

            # save 32 time sequence data for each sample row 
            if self.Q32.qsize() == 31:
                self.Q32.put(row)
                data = [self.Q32.get() for i in range(32)]
                
                # save to row
                row32 = []
                for item in data: 
                    row32 += item
                with open(self.filename_32, 'a') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row32)

                # put back to the queue
                for item in data[1:]:
                    self.Q32.put(item)
            else: 
                self.Q32.put(row)

            # save 48 time sequence data for each sample row 
            if self.Q48.qsize() == 47:
                self.Q48.put(row)
                data = [self.Q48.get() for i in range(48)]
                
                # save to row
                row48 = []
                for item in data: 
                    row48 += item
                with open(self.filename_48, 'a') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row48)

                # put back to the queue
                for item in data[1:]:
                    self.Q48.put(item)
            else: 
                self.Q48.put(row)

    def shutdown(self):
        
        rospy.loginfo("The node is shutdown!")

if __name__ == "__main__":
    try:
        cloud2file()
        rospy.spin()
    except:
        rospy.loginfo("The node is terminated!")