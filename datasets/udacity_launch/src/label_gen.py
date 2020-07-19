'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.1.0
@Date: 2020-05-19 09:54:23
@LastEditTime: 2020-06-03 11:24:46
'''

"""
!!! Warning !!!: This is not a ROS node.

Pipeline:
1. load data from interpolated.csv and separate them by frame_id.
  - index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt
2. loading timestamps of all point data and registering it with image timestamps. 
  - scale to same unit: s
  - index = (np.argmin(abs(stamp1-stamp2[i]))
  - define tolerance
3. combine them by index and save to new file

Hint:
1. For point name: 1479425441 179272 us; 10Hz
2. For camera: 1479425441 209768182 ns; 20Hz
3. For interpolate.csv: steer: 50hz

tol: 0.05; about 0.02
"""

import os 
import glob
import numpy as np 
import pandas as pd

def load_point_stamp(data_path):
	""" loading time stamps of all point data file
	"""
	file_name = glob.glob(os.path.join(data_path,'*.pcd'))
	file_name.sort()
	stamps = []
	for i in range(len(file_name)):
		stamps.append(float(file_name[i].split('/')[-1].split('.')[0])/1e6) # to seconds

	print('Thera are {} point data file'.format(len(stamps)))

	return stamps


def labelGen(data_path, stamps_point):
	"""
	"""
	# load data
	print('== Loading data from {} =='.format(data_path))
	data = pd.read_csv(data_path)
	print('Original Data: {}, {}'.format(data.shape,data.columns))
    
	# seperate file
	frame_id = data['frame_id'].values
	center_idx, right_idx, left_idx = [], [], []
	for i in range(len(frame_id)):
		if frame_id[i] == 'center_camera':
			center_idx.append(i)
		elif frame_id[i] == 'right_camera':
			right_idx.append(i)
		elif frame_id[i] == 'left_camera':
			left_idx.append(i)
		else:
			raise NameError
	
	data_center = data.iloc[center_idx].values
	data_right  = data.iloc[right_idx].values
	data_left   = data.iloc[left_idx].values
	print('Center: {}; Right: {}; Left: {}'.format(data_center.shape, data_right.shape, data_left.shape))

	# register
	data_center_new = np.zeros((0,13)) # empty array
	data_right_new  = np.zeros((0,13))
	data_left_new   = np.zeros((0,13))
	current_k = -1
	tol = 0.05
	for i in range(len(stamps_point)):
		point_file = str(int(stamps_point[i]*1e6)) + '.pcd'

		diff_center = abs(data_center[:,1]/1e9 - stamps_point[i])
		if np.min(diff_center) < tol:
			center_k = np.argmin(diff_center)
			center_new = np.hstack([data_center[center_k], point_file])
			data_center_new = np.vstack([data_center_new, center_new])
            # ensure the time is continues
			if center_k > current_k:
				current_k = center_k
			else:
				print("Time is not continues!!")
		
		diff_right = abs(data_right[:,1]/1e9 - stamps_point[i])
		if np.min(diff_right) < tol:
			right_k  = np.argmin(diff_right)
			right_new  = np.hstack([data_right[right_k], point_file])
			data_right_new  = np.vstack([data_right_new, right_new])
		
		diff_left = abs(data_left[:,1]/1e9 - stamps_point[i])
		if np.min(diff_left) < tol:
			left_k   = np.argmin(diff_left)
			left_new   = np.hstack([data_left[left_k], point_file])
			data_left_new = np.vstack([data_left_new, left_new])

		

	print("New files:", data_center_new.shape, data_right_new.shape, data_left_new.shape)
    
	# save
	base_dir = os.path.dirname(data_path)
	center_file = os.path.join(base_dir,'center.csv')
	right_file = os.path.join(base_dir,'right.csv')
	left_file = os.path.join(base_dir,'left.csv')
	
	columns = ['index', 'timestamp', 'width', 'height', 'frame_id', 'filename',
               'angle', 'torque', 'speed', 'lat', 'long', 'alt', 'point_filename']
	pd.DataFrame(data_center_new, columns=columns).to_csv(center_file)
	pd.DataFrame(data_right_new, columns=columns).to_csv(right_file)
	pd.DataFrame(data_left_new, columns=columns).to_csv(left_file)


	

if __name__ == '__main__':

	data_path = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_002/HMB_6/interpolated.csv'
	point_path = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_002/HMB_6/points'

	stamps_point = load_point_stamp(point_path)
	labelGen(data_path, stamps_point)
	
  