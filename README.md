# aruco_detection
A ROS package for detecting aruco markers

## How to Run
firstly, put this repository into the catkin workspace and build the package.

Then you can run:

```bash
rosrun aruco_detection aruco_detector.py
```

## Parameters to Modify

All modified parameters are in the script [aruco_detector.py](scripts/aruco_detector.py)

1. Aruco marker size: change `marker_length` at line [#14](https://github.com/JackHaoyingZhou/aruco_detection/blob/main/scripts/aruco_detector.py#L14), the unit is meter.
2. Enable Visualization: change `show_image` to be True at line [#14](https://github.com/JackHaoyingZhou/aruco_detection/blob/main/scripts/aruco_detector.py#L14)