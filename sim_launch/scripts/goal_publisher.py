#!/usr/bin/env python
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import numpy as np

rospy.init_node('goal_publisher')
rospy.sleep(.1)

client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
client.wait_for_server()
goal = MoveBaseGoal()
goal.target_pose.pose.orientation.w = 1.0
goal.target_pose.pose.position.y = 100.0*np.random.rand()
goal.target_pose.pose.position.x = 100.0*np.random.rand()
goal.target_pose.header.frame_id = 'world'

client.send_goal(goal)
client.wait_for_result()
print(client.get_result())
print(client.get_state())
