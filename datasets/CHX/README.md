<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-05-17 20:02:12
 * @LastEditTime: 2020-07-19 16:09:42
--> 
## CHX_001
| Date | Lighting Conditions | Duration | Compressed Size | Direct Download | Torrent | MD5 |
| ---- | :------------------:| --------:| ---------------:|:---------------:|:-------:|:---:|
| 11/18/2016 | Daytime | --- | 1.4 GB | None | [Torrent](https://github.com/udacity/self-driving-car/blob/master/datasets/CHX/CHX_001.tar.gz.torrent) | df85994915c0b650332f8c187c4f178b |

Lap around block at Udacity office with **new HDL-32E LIDAR** from George Hotz. Can almost create a loop-closure using NDT mapping, more sophisticated methods of SLAM should be able to create a high-fidelity map

### Overview
After decompressing the CHX_001.tar.gz, we get the following files:
```python
CHX_001
├── bin_0.400000_cal_loop.pcd
├── cal_loop.bag
├── cal_loop.png
└── cal_loop.txt
```

The we read and dump data from the rosbag using the tools provided by <https://github.com/rwightman/udacity-driving-reader>, and save them to the `CHX_001_output` folder.
```python
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
- `6311`, `6288` and  `6311` .jpg images captured by the center, left, and right camera saved in the `center`, `left` and `right` folder, respectively. The name of the images is the timestamp. (These images are captured continously.)
- From `cal_loop.yaml` we can know that the data is collected start at 1479512770295936000 (ns), end at 1479513085906092800 and the duration is 315610157000.
- For the camera.csv, there are totally `18910` file logs. about `20 FPS` 
  ```python
  timestamp,width,height,frame_id,filename
  1479512770308802490,640,480,center_camera,center/1479512770308802490.jpg
  1479512770321979743,640,480,right_camera,right/1479512770321979743.jpg
  1479512770328891343,640,480,left_camera,left/1479512770328891343.jpg
  1479512770358764162,640,480,center_camera,center/1479512770358764162.jpg
  1479512770372019250,640,480,right_camera,right/1479512770372019250.jpg
  ```
 - For the interpolated.csv, there is the time-synchronized data:
   ```python
    index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt
    2016-11-18 23:46:10.308802490,1479512770308802490,640,480,center_camera,center/1479512770308802490.jpg,0.0,0.0,0.0,0.0,0.0,0.0
    2016-11-18 23:46:10.321979743,1479512770321979743,640,480,right_camera,right/1479512770321979743.jpg,0.0,0.0,0.0,37.39973449707031,-122.108154296875,-8.727110528797837
    2016-11-18 23:46:10.328891343,1479512770328891343,640,480,left_camera,left/1479512770328891343.jpg,0.0,0.0,0.0,37.39973449707031,-122.108154296875,-8.736923517877008
    2016-11-18 23:46:10.358764162,1479512770358764162,640,480,center_camera,center/1479512770358764162.jpg,0.12566371262073517,-0.125,0.0,37.39973449707031,-122.108154296875,-8.734223107155499
    2016-11-18 23:46:10.372019250,1479512770372019250,640,480,right_camera,right/1479512770372019250.jpg,0.12566371262073517,-0.125,0.0,37.39973449707031,-122.108154296875,-8.732458351780416
   ```
 - For the steering.csv
   ```python
    timestamp,angle,torque,speed
    1479512770330361947,0.12566371262073517,-0.125,0.0
    1479512770350364851,0.12566371262073517,-0.125,0.0
    1479512770370505580,0.12566371262073517,-0.125,0.0
    1479512770390505431,0.12566371262073517,-0.125,0.0
    1479512770410505270,0.12566371262073517,-0.125,0.0
   ```
