#!/usr/bin/env python
import rospy
import rosbag
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from nav_msgs.msg import OccupancyGrid
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
        self.octomap_launch = None
        self.status = None
        self.status_gt = None
        self.gt_occ_grid = None
        rospy.Subscriber("move_base/status", GoalStatusArray, self.status_callback)
        rospy.Subscriber("ground_truth_planning/move_base/status", GoalStatusArray, self.gt_status_callback)
        rospy.Subscriber("ground_truth_planning/map", OccupancyGrid, self.occ_grid_callback)

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

    def occ_grid_callback(self, msg):
        self.gt_occ_grid = msg

    def isOccupied(self, metric_point):
        info = self.gt_occ_grid.info
        origin = np.empty_like(metric_point)
        origin[0] = info.origin.position.x
        origin[1] = info.origin.position.y
        ind = np.floor((metric_point - origin)/info.resolution).astype('int')
        grid = np.array(self.gt_occ_grid.data).reshape(info.height, info.width)
        if grid[ind[1], ind[0]] > 50:
            return True
        return False

    def get_unoccupied_point(self, origin, radius):
        occupied = True
        point = None
        while not rospy.is_shutdown() and occupied:
            point = origin + radius * np.random.random_sample(origin.shape)
            occupied = self.isOccupied(point)
        return point

    def generate_data(self):
        world_size = np.array(rospy.get_param('~world_size', [100., 100.]))
        world_origin = np.array(rospy.get_param('~world_origin', [0., 0.]))
        goal_distance = rospy.get_param('~goal_distance', 30)
        timeout = rospy.get_param('~timeout', 60)
        ros_pack = rospkg.RosPack()
        sim_launch_path = ros_pack.get_path('sim_launch')
        bag_name = sim_launch_path + '/data/' + time.strftime("%Y%m%d-%H%M%S") + '.bag'

        # launch unity
        binary_path = ros_pack.get_path('sim_launch') + '/sim_binary/arl_sim_binary.x86_64'
        with open(os.devnull, 'w') as fp:
            self.proc = subprocess.Popen([binary_path], stdout=fp)

        # roslaunch octomap
        uuid2 = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid2)
        sim_launch_launch_path = sim_launch_path + '/launch/gt_octomap_only.launch'
        roslaunch_file2 = [(roslaunch.rlutil.resolve_launch_arguments([sim_launch_launch_path])[0], [])]
        self.octomap_launch = roslaunch.parent.ROSLaunchParent(uuid2, roslaunch_file2)
        self.octomap_launch.start()
        rospy.logwarn("started octomap launch")

        rospy.sleep(3)

        # generate random start points
        start_pt = self.get_unoccupied_point(world_origin, (world_size - goal_distance))
        yaw = np.random.rand()*2*np.pi
        rospy.logwarn('start location ' + str(start_pt[0]) + ',' + str(start_pt[1]))

        # roslaunch sim_launch
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        sim_launch_launch_path = sim_launch_path + '/launch/generate_data_bags_helper.launch'
        cli_args = [sim_launch_launch_path, 'x:=' + str(start_pt[0]), 'y:=' + str(start_pt[1]), 'yaw:=' + str(yaw), 'bag_name:=' + bag_name]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        self.launches = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        self.launches.start()
        rospy.logwarn("started everything else launch")

        # Call action server
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client_gt = actionlib.SimpleActionClient('ground_truth_planning/move_base', MoveBaseAction)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and not client.wait_for_server(rospy.Duration(1)):
            rospy.logwarn("NO ACTION SERVER - waiting")
            rospy.sleep(.1)
            if rospy.Time.now() - start_time > rospy.Duration(30):
                rospy.logerr("Timeout on action server")
                return

        while not rospy.is_shutdown() and not client_gt.wait_for_server(rospy.Duration(1)):
            rospy.logwarn("NO ground truth ACTION SERVER - waiting")
            rospy.sleep(.1)
            if rospy.Time.now() - start_time > rospy.Duration(30):
                rospy.logerr("Timeout on action server")
                return

        goal = MoveBaseGoal()
        # TODO also randomize orientation
        goal_pt = self.get_unoccupied_point(start_pt, goal_distance)  # generate random goal point
        goal.target_pose.pose.orientation.w = 1
        goal.target_pose.pose.position.x = goal_pt[0]
        goal.target_pose.pose.position.y = goal_pt[1]
        rospy.logwarn('goal location ' + str(goal.target_pose.pose.position.x) + ',' + str(goal.target_pose.pose.position.y))

        goal.target_pose.header.frame_id = 'world'
        client.send_goal(goal)
        client_gt.send_goal(goal)
        rospy.loginfo("Sent goal")
        rospy.logwarn("Waiting for Action Server result " + str(timeout) + "s")
        start_time = rospy.Time.now()
        result = False
        while not rospy.is_shutdown():
            dur = rospy.Time.now() - start_time
            if self.status == GoalStatus.SUCCEEDED:
                result = True
                rospy.logwarn("Action server succeeded")
                break
            if 2 <= self.status <= 5 or 2 <= self.status_gt <= 5:
                if dur < rospy.Duration(10):
                    result = False
                    rospy.logerr("Action server Failed")
                    break
                else:
                    rospy.logwarn("Action server Failed but not deleting bag.")
                    result = True
                    break
            if dur > rospy.Duration(timeout):
                result = True
                rospy.logerr("Timeout, not deleting bag")
                break
            rospy.sleep(.1)
        if not result:
            rospy.logwarn("Deleting bag")
            if os.path.exists(bag_name):
                os.remove(bag_name)
            if os.path.exists(bag_name + '.active'):
                os.remove(bag_name + '.active')

    def shutdown(self):
        try:
            self.octomap_launch.shutdown()
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

    num_iterations = rospy.get_param('~num_iterations', 10)
    gd = GenerateData()
    for i in range(num_iterations):
        if not rospy.is_shutdown():
            gd.generate_data()
        gd.shutdown()
