#!/usr/bin/env python
import rospy
import rosbag
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
import roslaunch
import rospkg
import numpy as np
import time
import subprocess
import os


class GenerateData:
    def __init__(self):
        self.proc = None
        self.launches = None
        self.status = -1
        self.status_gt = None
        rospy.Subscriber("move_base/status", GoalStatusArray, self.status_callback)
        rospy.Subscriber("ground_truth_planning/move_base/status", GoalStatusArray, self.gt_status_callback)


    def status_callback(self, msg):
        if len(msg.status_list) > 0:
            self.status = msg.status_list[0].status
        else:
            self.status = -1

    def gt_status_callback(self, msg):
        if len(msg.status_list) > 0:
            self.status_gt = msg.status_list[0].status
        else:
            self.status_gt = -1

    def generate_data(self,):

        # TODO make this launch file args
        world_size = [100., 100.]
        world_origin = [0., 0.]
        goal_distance = 30
        timeout = 60

        ros_pack = rospkg.RosPack()
        sim_launch_path = ros_pack.get_path('sim_launch')
        bag_name = sim_launch_path + '/data/' + time.strftime("%Y%m%d-%H%M%S") + '.bag'

        # generate random start points
        start_x = world_origin[0] + world_size[0]*np.random.rand()
        start_y = world_origin[1] + world_size[0]*np.random.rand()
        rospy.logwarn('start location ' + str(start_x) + ',' + str(start_y))

        # launch unity
        binary_path = ros_pack.get_path('sim_launch') + '/sim_binary/arl_sim_binary.x86_64'
        with open(os.devnull, 'w') as fp:
            self.proc = subprocess.Popen([binary_path], stdout=fp)
        rospy.sleep(3)

        # roslaunch sim_launch
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        sim_launch_launch_path = sim_launch_path + '/launch/generate_data_bags_helper.launch'
        cli_args = [sim_launch_launch_path, 'x:=' + str(start_x), 'y:=' + str(start_y), 'bag_name:=' + bag_name]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        self.launches = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        self.launches.start()
        rospy.logwarn("started launch")

        # Call action server
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        while not rospy.is_shutdown() and self.status is None or not client.wait_for_server(rospy.Duration(1)):
            rospy.logerr("NO ACTION SERVER - waiting")
            rospy.sleep(.1)
        while not rospy.is_shutdown() and self.status_gt is None:
            rospy.logerr("NO ground truth ACTION SERVER - waiting")
            rospy.sleep(.1)
        goal = MoveBaseGoal()
        # TODO also randomize orientation
        goal.target_pose.pose.orientation.w = 1
        goal.target_pose.pose.position.x = start_x + goal_distance*np.random.rand()
        goal.target_pose.pose.position.y = start_y + goal_distance*np.random.rand()  # generate random goal point
        rospy.logwarn('goal location ' + str(goal.target_pose.pose.position.x) + ',' + str(goal.target_pose.pose.position.y))

        goal.target_pose.header.frame_id = 'world'
        client.send_goal(goal)
        rospy.loginfo("Sent goal")
        rospy.logwarn("Waiting for Action Server result " + str(timeout) + "s")
        start_time = rospy.Time.now()
        result = False
        while not rospy.is_shutdown():
            if self.status == GoalStatus.SUCCEEDED:
                result = True
                break
            if 2 <= self.status <= 5 or 2 <= self.status_gt <= 5:
                result = False
                rospy.logerr("Action Client Failed")
                break
            if rospy.Time.now() - start_time > rospy.Duration(timeout):
                result = False
                rospy.logerr("Timeout")
                break
            rospy.sleep(.1)
        if result:
            rospy.logwarn("Action server succeeded")
        else:
            rospy.logwarn("Action server failed, deleting bag")
            if os.path.exists(bag_name):
                os.remove(bag_name)
            if os.path.exists(bag_name + '.active'):
                os.remove(bag_name + '.active')

    def shutdown(self):
        try:
            self.launches.shutdown()
            rospy.sleep(5)
        except:
            rospy.logwarn("couldn't shut down roslaunches")
        try:
            self.proc.terminate()
        except:
            rospy.logwarn("couldn't shut down unity")


if __name__ == '__main__':
    rospy.init_node('generate_data_bags', anonymous=True)

    num_iterations = 10  # TODO make launch file param
    gd = GenerateData()
    for i in range(num_iterations):
        if not rospy.is_shutdown():
            gd.generate_data()
        gd.shutdown()
