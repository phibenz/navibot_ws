#!/usr/bin/env python
import rospy
#import hokuyo_node
import sensor_msgs.msg
from gazebo_msgs.msg import ContactsState

def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s", data.states[0].collision2_name)
    #for i in range(len(data.states)):
    #	print("###########Entry ",i,"###############")
    #	print(data.states[i])

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/navibot left_wheel_bumper", ContactsState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()