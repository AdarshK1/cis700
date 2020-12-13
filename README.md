# cis700
Fusion of Learning-based Navigation with Classical Planning. 

This work is a collaboration with [Ian Miller](https://github.com/iandouglas96?tab=repositories) and [Laura Jarin-Lipschitz](https://github.com/ljarin). As part of CIS700 with Professor Dinesh Jayaraman at Penn GRASP.

Our proposed method combines the long-term and long-range memory of traditional SLAM and global planners while benefiting from the increased semantic agility of a learned system. This
repository contains ROS related and pytorch work for running a Unity simulator with a Clearpath Husky for data collection
of a ground truth planner that is aware of "flexible/traversable" obstacles as well as a naive one.

### Packages

https://www.patrickmin.com/binvox/

`sudo apt-get install octomap-tools`


### Launching Sim
* unity binary or editor
```
roslaunch arl_unity_ros_ground simulator_with_husky_nolidar.launch
roslaunch sim_launch sim_with_ground_truth.launch 
```
