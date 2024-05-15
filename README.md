# aruco_detection
A ROS package for detecting aruco markers

## How to Run
firstly, put this repository into the catkin workspace and build the package.

Then you can run:

```bash
rosrun aruco_detection aruco_detector.py
```

If you want to change the marker length, please change the number at line #13 in aruco_detector.py, the unit is meter.