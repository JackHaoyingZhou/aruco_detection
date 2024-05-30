# aruco_detection
A ROS package for detecting aruco markers

## How to Run
firstly, put this repository into the catkin workspace and build the package.

Then you can run:

```bash
rosrun aruco_detection aruco_detector.py
```

## Parameter Definition

All modified parameters are in the script [aruco_detector.py](scripts/aruco_detector.py), you can set your own default values at line [#106-#132](https://github.com/JackHaoyingZhou/aruco_detection/blob/main/scripts/aruco_detector.py#L106)

| argument | argument meaning                                                       | default value |
|----------|------------------------------------------------------------------------|---------------|
| -c       | ROS namespace for the camera                                           | /depstech     |
| -m       | the size of the Aruco markers, the unit is meter                       | 0.03          |
| -t       | the Aruco marker style, `1` represents 6x6_250, `2` stands for 4x4_50  | 1             |
| -i       | add this argument if you want to show the images when detection        | False         |
