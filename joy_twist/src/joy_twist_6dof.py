#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class JoyTwist:
    def joy_cb(self, joy):
        self.joy = joy
        self.eps = 0.1

    def send_cmds(self, time):
        if self.joy is None:
            return

        do_move = False
        if abs(self.joy.axes[0]) > self.eps or \
           abs(self.joy.axes[1]) > self.eps or \
           self.joy.axes[2] < 1-self.eps or \
           abs(self.joy.axes[3]) > self.eps or \
           abs(self.joy.axes[4]) > self.eps or \
           self.joy.axes[5] < 1-self.eps or \
           abs(self.joy.buttons[5]) > 0 or \
           abs(self.joy.buttons[4]) > 0:
            do_move = True

        twist = Twist()
        if do_move:
            twist.linear.x = 0.2*((1-self.joy.axes[5]) - (1-self.joy.axes[2]))/2
            twist.linear.y = 0.2*self.joy.axes[0]
            twist.linear.z = 0.2*self.joy.axes[1]
            twist.angular.x = 0.2*(self.joy.buttons[5] - self.joy.buttons[4])
            twist.angular.y = 0.2*self.joy.axes[4]
            twist.angular.z = 0.2*self.joy.axes[3]

        self.twist_pub_.publish(twist)

    def run(self):
        rospy.init_node("JoyTwist")
        joy_sub = rospy.Subscriber("joy", Joy, callback=self.joy_cb)
        self.twist_pub_ = rospy.Publisher("twist", Twist, queue_size=1)

        self.joy = None
        self.timer = rospy.Timer(rospy.Duration(0.05), self.send_cmds)
        
        rospy.spin()

if __name__ == '__main__':
    jt = JoyTwist()
    jt.run()
