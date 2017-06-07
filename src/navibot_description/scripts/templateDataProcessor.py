import rospy
import time
import numpy as np
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from gazebo_msgs.msg import ModelStates
from control_msgs.msg import JointControllerState

class dataProcessor:
	def __init__(self):
		# TODO check if ros, gazebo and tos_control is started

		rospy.init_node('dataProcessor', anonymous=True)

		data=rospy.wait_for_message("/navibot/laser/scan", LaserScan)
		state=rospy.wait_for_message("/gazebo/model_states", ModelStates)
		print(state.pose[2])

		self.robotName = 'navibot'
		self.updatesPerStep=100 #This parameter should be shared for environmentControl and dataProcessor
		self.stateSize=4
		self.deltaStep=int(self.updatesPerStep/self.stateSize)

		self.SenorRangeMax=100. #TODO: Needs clarification
		self.SensorRangeMin=0.

		self.numSensorVal=7 # TODO maybe as input
		self.state=np.zeros((self.stateSize, self.numSensorVal))

		# Get Robot Index to identify in modelStates
		self.rIndexSub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_robotIndex)
		time.sleep(0.02) # Give some time to process the index in callback_robotIndex
		self.rIndexSub.unregister()

		self.laserRangeSub = rospy.Subscriber("/navibot/laser/scan", LaserScan, self.callback_sensor)
		self.laserRangePub = rospy.Publisher("/output/laser_ranges", numpy_msg(Floats), queue_size=10)
		
		self.positionSub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_position)
		self.positionPub = rospy.Publisher("/output/position", numpy_msg(Floats), queue_size=10)

		#self.stateSub = rospy.Subscriber("/output/position", numpy_msg(Floats), self.callback_state)
		#self.statePub = rospy.Publisher("/output/state", numpy_msg(Floats), queue_size=10)
		
		#self.leftWheelVelSub = rospy.Subscriber("/navibot/left_wheel_hinge_velocity_controller/state", JointControllerState, self.callback_leftWheelVel)
		self.leftWheelVelPub = rospy.Publisher("/navibot/left_wheel_hinge_velocity_controller/command", 
												Float64, queue_size=10)
		rospy.spin()
		

	def callback_robotIndex(self, modelStateData):
		robotExists = False
		for i in range(len(modelStateData.name)): 
			if modelStateData.name[i] == self.robotName:
				self.robotIndex = i
				robotExists = True
		if not robotExists:
			raise ROSException('Robot with name {} not found'.format(self.robotName))

	def callback_sensor(self, laserData):
		# Returns pure laser data
		laserRanges=np.array(laserData.ranges)
		self.laserRangePub.publish(laserRanges)

	def callback_position(self, modelStateData):
		robotPosition=np.zeros((1,3))
		robotPosition[0,0]=modelStateData.pose[self.robotIndex].position.x
		robotPosition[0,1]=modelStateData.pose[self.robotIndex].position.y
		robotPosition[0,2]=modelStateData.pose[self.robotIndex].position.z
		self.positionPub.publish(robotPosition)

	def callback_state(self, stateData):
		print(stateData)
		# Normalize sensor data
		stateData=stateData-self.SensorRangeMin/(self.SenorRangeMax-self.SensorRangeMin)

if __name__ == '__main__':
	dP=dataProcessor()
	