'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-30 22:01:17
@LastEditTime: 2020-07-19 16:10:48
@Description:  Some useful codes collected from champion schemes.
'''

######################## I. NVIDIA AutoPilot Net ########################
# ref: https://github.com/SullyChen/Autopilot-TensorFlow

## Load Data
# A direct way for loading data
import cv2
import random
import numpy as np

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * 3.14159265 / 180) ## use radis

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], (200, 66)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out



## Model Design
#scale the atan output
y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) 

## Prediction
# Steering Angle Visulaization
img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    if not windows:
        call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", full_image)
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()


######################## II. Autumn Scheme ########################

## data processing
# covert 7 successive images into an averaged optical flow image
last = []
prev_image = None

for i, filename in enumerate(files):
    img = scipy.misc.imread(args.data_dir + '/' + files[i] + input_type, mode='RGB')
    prev = prev_image if prev_image is not None else img
    prev_image = img
    prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    last.append(flow)
    if len(last) > args.window_size:
        last.pop(0)

    avg_flow = sum(last) / len(last)
    mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

    hsv = np.zeros_like(prev_image)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if args.show_image:
        cv2.imshow('flow', bgr)
        cv2.waitKey(1)

    if not os.path.exists(args.data_dir + '/' + args.output_dir):
        os.makedirs(args.data_dir + '/' + args.output_dir)
    cv2.imwrite(args.data_dir + '/' + args.output_dir + '/' + files[i] + '.png', bgr)
    print('Saving to ' + args.data_dir + '/' + args.output_dir + '/' + files[i] + '.png')

## Data reader
# balancing steering label distribution

DATA_DIR = '/vol/data/'
FILE_EXT = '.png'


class DataReader(object):
    def __init__(self, data_dir=DATA_DIR, file_ext=FILE_EXT, sequential=False):
        self.load()

    def load(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        total = 0
        count01 = count005 = count002 = count0 = 0

        with open('interpolated_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # balance the data distribution (crop angle data near 0 )
                angle = float(row['steering_angle'])
                if angle > 0.1 or angle < -0.1 and random.random() > 0.2:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count01 += 1
                elif (angle > 0.05 or angle < -0.5) and random.random() > 0.2:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count005 += 1
                elif (angle > 0.02 or angle < -0.02) and random.random() > 0.7:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count002 += 1
                elif random.random() > 0.8:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count0 += 1
                total += 1

        with open('train_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                angle = float(row['steering_angle'])
                xs.append(DATA_DIR + 'Ch2_Train/center/flow_7_local/' + row['frame_id'] + FILE_EXT)
                ys.append(row['steering_angle'])
                total += 1

        print('> 0.1 or < -0.1: ' + str(count01))
        print('> 0.05 or < -0.05: ' + str(count005))
        print('> 0.02 or < -0.02: ' + str(count002))
        print('~0: ' + str(count0))
        print('Total data: ' + str(total))

        self.num_images = len(xs)

        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(xs) * 0.8)]

        self.val_xs = xs[-int(len(xs) * 0.2):]
        self.val_ys = ys[-int(len(xs) * 0.2):]

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)

    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out

    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.val_xs[(self.val_batch_pointer + i) % self.num_val_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return x_out, y_out

    def load_seq(self):
        xxxx




######################## III. cg23 Scheme ########################

## Load data
def camera_adjust(angle,speed,camera):

    # Left camera -20 inches, right camera +20 inches (x-direction)
    # Steering should be correction + current steering for center camera

    # Chose a constant speed
    speed = 10.0  # Speed

    # Reaction time - Time to return to center
    # The literature seems to prefer 2.0s (probably really depends on speed)
    if speed < 1.0:
        reaction_time = 0
        angle = angle
    else:
        reaction_time = 2.0 # Seconds

        # Trig to find angle to steer to get to center of lane in 2s
        opposite = 20.0 # inches
        adjacent = speed*reaction_time*12.0 # inches (ft/s)*s*(12 in/ft) = inches (y-direction)
        angle_adj = np.arctan(float(opposite)/adjacent) # radians
    	
        # Adjust based on camera being used and steering angle for center camera
        if camera == 'left':
            angle_adj = -angle_adj
        angle = angle_adj + angle

    return angle


#### 
'''
Results video generator Udacity Challenge 2
Original By: Comma.ai Revd: Chris Gundling
'''

from __future__ import print_function

import argparse
import sys
import numpy as np
import h5py
import pygame
import json
import pandas as pd
from os import path
#from keras.models import model_from_json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab
from pygame.locals import *

from data_TS import *

pygame.init()
size = (320*2, 160*3)
#size2 = (640,160)
pygame.display.set_caption("epoch data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.set_alpha(None)

#camera_surface = pygame.surface.Surface((320,160),0,24).convert()
camera_surface = pygame.surface.Surface((320,160),0,24).convert()
clock = pygame.time.Clock()

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[row-sz:row+sz, col-sz:col+sz] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) #* deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color)

# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
    parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    args = parser.parse_args()

    dataset_path = args.dataset
    image_size = (args.resized_image_width, args.resized_image_height)
    camera = args.camera

    # steerings and images
    steering_log = path.join(dataset_path, 'final_example.csv')
    image_log = path.join(dataset_path, 'camera.csv')
    camera_images = dataset_path

    df_test = pd.read_csv('epoch.csv',usecols=['frame_id','steering_angle'],index_col = None)
    df_truth = pd.read_csv('ch2_final_eval.csv',usecols=['frame_id','steering_angle'],index_col = None)
    
    # Testing on the Test Images
    test_generator = data_generator(steering_log=steering_log,
                         image_log=image_log,
                         image_folder=camera_images,
                         camera=camera,
                         batch_size=5614,
                         image_size=image_size,
                         timestamp_start=14794254411,
                         timestamp_end=14794257218,
                         shuffle=False,
                         preprocess_input=normalize_input,
                         preprocess_output=exact_output)

    print('Made it to Testing')

    test_x, test_y = test_generator.next()
    print('test data shape:', test_x.shape)
    
    # Create second screen with matplotlib
    fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    #ax.legend(loc='upper left',fontsize=8)
    line1, = ax.plot([], [],'b.-',label='Human')
    line2, = ax.plot([], [],'r.-',label='Model')
    A = []
    B = []
    ax.legend(loc='upper left',fontsize=8)
    
    red=(255,0,0)
    blue=(0,0,255)
    myFont = pygame.font.SysFont("monospace", 18)
    randNumLabel = myFont.render('Human Steer Angle:', 1, blue)
    randNumLabel2 = myFont.render('Model Steer Angle:', 1, red)
    speed_ms = 5 #log['speed'][i]

    # Run through all images
    for i in range(5614):
        #if i%100 == 0:
        #    print('%.2f seconds elapsed' % (i/20))
        img = test_x[i,:,:,:].swapaxes(0,2).swapaxes(0,1)

        predicted_steers = df_test['steering_angle'].loc[i]
        actual_steers = df_truth['steering_angle'].loc[i]

        draw_path_on(img, speed_ms, actual_steers/5.0)
        draw_path_on(img, speed_ms, predicted_steers/5.0, (255, 0, 0))

        A.append(df_test['steering_angle'].loc[i])
        B.append(df_truth['steering_angle'].loc[i])
        line1.set_ydata(A)
        line1.set_xdata(range(len(A)))
        line2.set_ydata(B)
        line2.set_xdata(range(len(B)))
        ax.relim()
        ax.autoscale_view()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        screen.blit(surf, (0,320))

        # draw on
        pygame.surfarray.blit_array(camera_surface, img.swapaxes(0,1))
        camera_surface_2x = pygame.transform.scale2x(camera_surface)
        screen.blit(camera_surface_2x, (0,0))
	
        diceDisplay = myFont.render(str(actual_steers*(180/np.pi)), 1, blue)
        diceDisplay2 = myFont.render(str(predicted_steers*(180/np.pi)), 1, red)
        screen.blit(randNumLabel, (50, 280))
        screen.blit(randNumLabel2, (400, 280))
        screen.blit(diceDisplay, (50, 300))
        screen.blit(diceDisplay2, (400, 300))
        clock.tick(60)
        pygame.display.flip()


############################# IV. chauffeur ####################################

def generate_video(submission_path,
                   images_path,
                   video_path,
                   temp_dir):

    assert video_path.endswith('.mp4'), 'h264 pls'
    safe_makedirs(os.path.dirname(video_path))

    filename_angles = []
    with open(submission_path) as f:
        for line in f:
            if "frame" not in line:
                ts, angle = line.strip().split(',')
                filename_angles.append((ts, angle))

    progress_bar = IncrementalBar(
        'Generating overlay',
        max=len(filename_angles),
        suffix='%(percent).1f%% - %(eta)ds')

    for filename, angle in filename_angles:
        img_path = os.path.join(images_path, filename + '.jpg')
        cv_image = overlay_angle(img_path, float(angle))
        cv2.imwrite(os.path.join(temp_dir, filename + '.png'), cv_image)
        progress_bar.next()

    print '\nGenerating mpg video'
    _, mpg_path = tempfile.mkstemp()
    subprocess.check_call([
        'mencoder',
        'mf://%s/*.png' % temp_dir,
        '-mf',
        'type=png:fps=20',
        '-o', mpg_path,
        '-speed', '1',
        '-ofps', '20',
        '-ovc', 'lavc',
        '-lavcopts', 'vcodec=mpeg2video:vbitrate=2500',
        '-oac', 'copy',
        '-of', 'mpeg'
    ], stdout=FNULL, stderr=subprocess.STDOUT)

    print 'Converting mpg video to mp4'
    try:
        subprocess.check_call([
            'ffmpeg',
            '-i', mpg_path,
            video_path
        ], stdout=FNULL, stderr=subprocess.STDOUT)
    finally:
        os.remove(mpg_path)

    print 'Wrote final overlay video to', video_path


def overlay_angle(img_path, angle):
    center=(320, 400)
    radius=50
    cv_image = cv2.imread(img_path)
    cv2.circle(cv_image, center, radius, (255, 255, 255), thickness=4, lineType=8)
    x, y = point_on_circle(center, radius, -angle)
    cv2.circle(cv_image, (x,y), 6, (255, 0, 0), thickness=6, lineType=8)
    cv2.putText(
        cv_image,
        'angle: %.5f' % get_degrees(angle),
        (50, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255))

    return cv_image


def get_degrees(radians):
    return (radians * 180.0) / 3.14


def point_on_circle(center, radius, angle):
    """ Finding the x,y coordinates on circle, based on given angle
    """
    # center of circle, angle in degree and radius of circle
    shift_angle = -3.14 / 2
    x = center[0] + (radius * cos(shift_angle + angle))
    y = center[1] + (radius * sin(shift_angle + angle))

    return int(x), int(y)


## visual cnn model 
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.models import load_model

# Only supports convolution filters
def visualize_cnn_model(model_file, image_file, output_dir):
  model = load_model(model_file)
  print "model json:"
  print model.to_json()

  X = np.load(open(image_file, 'r'))

  #normalize image
  X = (X - (255.0/2))/255.0
  X = np.expand_dims(X, axis=0)

  num_layers = len(model.layers)

  for i in range(1, num_layers-1):
    if 'convolution' in model.layers[i].name:
      get_nth_layer_output = K.function([model.layers[0].input], [model.layers[i].output])
      layer_output = get_nth_layer_output([X])[0]
      print model.layers[i].name
      print layer_output.shape
      fig = plt.figure(figsize=(30.0, 15.0))
      for filter_num in range(0, layer_output.shape[3]):
        fig.add_subplot(layer_output.shape[3]/8, 8, filter_num)
        plt.imshow(layer_output[0, :, :, filter_num])
      plt.savefig(os.path.join(output_dir, model.layers[i].name + ".png"))

if __name__ == '__main__':
  if (len(sys.argv) <= 3):
     print "python visualize_cnn_model.py <model.h5 file> <image.jpg.npy file> <output dir>"
     sys.exit()
  model_file = sys.argv[1]
  image_file = sys.argv[2]
  output_file = sys.argv[3]
  visualize_cnn_model(model_file, image_file, output_file)

################################# V. rambo ##################################
# computed lag 1 differences between frames and used 2 consecutive differenced images.
# For example, at time t we used [x_{t} - x_{t-1}, x_{t-1} - x_{t-2}] as input where x corresponds to the grayscale image. 

## input data
def make_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print "Processed " + str(i) + " images..."
        for j in range(num_channels):
            path0 = df['fullpath'].iloc[i - j - 1]
            path1 = df['fullpath'].iloc[i - j]
            img0 = load_img(data_path + path0, grayscale=True, target_size=(row, col))
            img1 = load_img(data_path + path1, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j] = img[:, :, 0]
    return X, np.array(df["angle"].iloc[num_channels:])

# 这种应该和光流信息是类似的；思考我们还可以使用哪些信息？图像边缘信息，先做个canny边缘检测；加上光流运动信息；最后结合lstm


import numpy as np
import pandas as pd
import pygame
import glob
from config import VisualizeConfig

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

config = VisualizeConfig()
preds = pd.read_csv(config.pred_path)
true = pd.read_csv(config.true_path)
filenames = glob.glob(config.img_path)

pygame.init()
size = (640, 320)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

for i in range(1000):
#for i in range(len(filenames)):
    angle = preds["steering_angle"].iloc[i] # radians
    true_angle = true["steering_angle"].iloc[i] # radians
    
    # add image to screen
    img = pygame.image.load(filenames[i])
    screen.blit(img, (0, 0))
    
    # add text
    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    screen.blit(pred_txt, (10, 280))
    screen.blit(true_txt, (10, 300))

    # draw steering wheel
    radius = 50
    pygame.draw.circle(screen, WHITE, [320, 300], radius, 2) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [320 + int(x), 300 - int(y)], 7)
    
    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5) 
    
    #pygame.display.update()
    pygame.display.flip()
    