import roslaunch
import rospy
from std_srvs.srv import Empty

rospy.ServiceProxy('/gazebo/unpause_physics', Empty)