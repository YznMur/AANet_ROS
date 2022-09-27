# 📝 aanet_stereo_matching_ros #
***

ROS package for stereo matching and point cloud projection using [AANet](https://arxiv.org/abs/2004.09548)

# Steps to run this code:


After extracting the zip file: download the model and the rosbag as descrived above and but them in data folder.
```
- cd /path/to/aanet_stereo_matching
- cd docker
- ./docker_build.sh
- After building docker "successfully!" :  ./docker_start.sh
- ./docker_into.sh
- inside docker : cd src/aanet_ros_node/
- cd ./scripts/aanet/nets/deform_conv && sudo bash build.sh (PASSWORD IS "user")
- cd .. (go back to catkin_ws)
- catkin_make (if it doesn't work run : source /opt/ros/noetic/setup.bash)
- source devel/setup.bash
- roslaunch aanet_stereo_matching_ros aanet_stereo_matching_ros_rviz_sample.launch
```
- open a new terminal and run : 
``` 
- rviz 
```




## Запуск ROS_bridge
***
Перед запуском необходимо установить ROS_bridge:
```
- sudo apt install ros-dashing-ros1-bridge
```
Shell 1
***
```
- source /opt/ros/melodic/setup.bash
- roscore
```
Shell 2
***
```
- source /opt/ros/melodic/setup.bash
- source /opt/ros/dashing/setup.bash
- export ROS_MASTER_URI=http://localhost:11311
- ros2 run ros1_bridge dynamic_bridge --bridge-all-1to2-topics
