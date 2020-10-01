#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class JoyTwist:
    def joy_cb(self, joy):
        self.joy = joy

    def send_cmds(self, time):
        if self.joy is None:
            return

        twist = Twist()
        if self.joy.axes[5] < 0:
            twist.linear.x = self.joy.axes[1]
            twist.linear.y = self.joy.axes[0]
            twist.angular.z = self.joy.axes[3]

            self.twist1_pub_.publish(twist)
        elif self.joy.axes[2] < 0:
            twist.linear.x = self.joy.axes[1]
            twist.linear.y = self.joy.axes[0]
            twist.angular.z = self.joy.axes[3]

            self.twist2_pub_.publish(twist)

    def run(self):
        rospy.init_node("JoyTwist")
        joy_sub = rospy.Subscriber("joy", Joy, callback=self.joy_cb)
        self.twist1_pub_ = rospy.Publisher("twist1", Twist, queue_size=1)
        self.twist2_pub_ = rospy.Publisher("twist2", Twist, queue_size=1)

        self.joy = None
        self.timer = rospy.Timer(rospy.Duration(0.05), self.send_cmds)
        
        rospy.spin()

if __name__ == '__main__':
    jt = JoyTwist()
    jt.run()
