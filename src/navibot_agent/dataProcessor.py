import rospy
import time
import numpy as np

from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates, ContactsState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
import tf

from std_srvs.srv import Empty


class dataProcessor:
	def __init__(self, environmentController, robotName, phiLength, stateSize, numSensorVal, sensorRangeMax, sensorRangeMin, vel, vel_curve, update_time, speed_up):
		self.envC=environmentController

		self.robotName = robotName
		self.robotIndex, self.goalIndex = self.getIndeces()
		self.update_time=update_time
		self.speed_up=speed_up

		self.phiLength=phiLength
		self.isGoal=False

		self.stateSize=stateSize # TODO maybe as input
		self.numSensorVal=numSensorVal
		self.SenorRangeMax=sensorRangeMax #TODO: Needs clarification
		self.SensorRangeMin=sensorRangeMin

		self.vel=vel
		self.vel_curve=vel_curve
		self.lastDistance= 10
		#self.leftWheelVelPub = rospy.Publisher("/navibot/left_wheel_hinge_velocity_controller/command", 
		#										Float64, queue_size=10)
		#self.rightWheelVelPub = rospy.Publisher("/navibot/right_wheel_hinge_velocity_controller/command", 
		#										Float64, queue_size=10)
		self.velocityPub = rospy.Publisher("/navibot/velocity", Twist, queue_size=10)

	def getIndeces(self):
		# Helper function
		try:
			modelNames=rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5).name
		except:
			raise rospy.exceptions.ROSException
		robotExists = False
		for i in range(len(modelNames)): 
			if modelNames[i] == self.robotName:
				robotIndex = i
				robotExists = True
		if not robotExists:
			raise ValueError('Robot with name {} not found'.format(self.robotName))
		
		goalExists = False
		for i in range(len(modelNames)): 
			if modelNames[i] == 'goal':
				goalIndex = i
				goalExists = True
		if not goalExists:
			raise ValueError('Goal not found')
		return robotIndex, goalIndex

	def getState(self):
		time.sleep(self.update_time/self.speed_up)
		state=np.zeros((1, self.stateSize))
		try:
			laserData=np.array(rospy.wait_for_message("/navibot/laser/scan", LaserScan, timeout=5).ranges)
			navibot_tf=rospy.wait_for_message("/tf", TFMessage, timeout=5)
		except:
			raise rospy.exceptions.ROSException
		laserData[np.where(np.isinf(laserData))[0]]=0.
		state[0,0:self.numSensorVal]=(laserData-self.SensorRangeMin)/(self.SenorRangeMax-self.SensorRangeMin)
		
		robotPosition,robotOrientation=self.getRobotPosOri()
		goalPosition=self.getGoalPos()

		distance=np.sqrt((robotPosition[0]-goalPosition[0])**2+(robotPosition[1]-goalPosition[1])**2) # hypothenuse
		state[0,7]=distance/np.sqrt(200)
		
		quaternion=(navibot_tf.transforms[0].transform.rotation.x,
					navibot_tf.transforms[0].transform.rotation.y,
					navibot_tf.transforms[0].transform.rotation.z,
					navibot_tf.transforms[0].transform.rotation.w)
		euler= tf.transformations.euler_from_quaternion(quaternion)
		euler=np.array(euler)
		#print(euler*180/np.pi)
		#print('Robot2World', euler[2]*180/np.pi)
		oppo=goalPosition[1]-robotPosition[1]
		adja=goalPosition[0]-robotPosition[0]
		if adja<0 and oppo<0:
			angle=-np.pi-np.arcsin(oppo/distance)
		elif adja<0 and oppo>0:
			angle=np.pi-np.arcsin(oppo/distance)
		else:
			angle=np.arcsin(oppo/distance)
		#print('Robot2Goal', angle*180/np.pi)
		#print('difference', (angle-euler[2])*180/np.pi)
		state[0,8]=angle-euler[2]/(2*np.pi) #Orientation difference 
		return state

	def getRobotPosOri(self):
		try:
			roboPO=rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5).pose[self.robotIndex]
		except:
			raise rospy.exceptions.ROSException

		robotPosition=np.array((roboPO.position.x, roboPO.position.y, roboPO.position.z))
		robotOrientation=np.array((roboPO.orientation.x, roboPO.orientation.y, roboPO.orientation.z, roboPO.orientation.w))
		return robotPosition, robotOrientation

	def getGoalPos(self):

		try:
			goalPose=rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5).pose[self.goalIndex]
		except:
			raise rospy.exceptions.ROSException

		goalPos=np.array((goalPose.position.x, goalPose.position.y, goalPose.position.z))
		return goalPos

	def isPaused(self):
		return self.envC.physicsProp_client.call().pause

	def getReward(self):
		try:
			leftWheelBump=rospy.wait_for_message("/navibot/left_wheel_bumper", ContactsState, timeout=5)
			rightWheelBump=rospy.wait_for_message("/navibot/right_wheel_bumper", ContactsState, timeout=5)
			chassisBump=rospy.wait_for_message("/navibot/chassis_bumper", ContactsState, timeout=5)
			navibot_tf=rospy.wait_for_message("/tf", TFMessage, timeout=5)
		except:
			raise rospy.exceptions.ROSException
		goalPosition=self.getGoalPos()
		robotPosition,robotOrientation=self.getRobotPosOri()
		#if len(frontBump.states)>0:
		#	print('Front Bumper collided')
		#	collisionReward = -1
		if len(np.where([leftWheelBump.states[i].collision2_name.split('::')[0] != 'ground_plane' for i in range(len(leftWheelBump.states))])[0])>0:
			#print('Left wheel collided with ', leftWheelBump.states[i].collision2_name.split('::')[0])
			collisionReward = -1
		elif len(np.where([rightWheelBump.states[i].collision2_name.split('::')[0] != 'ground_plane' for i in range(len(rightWheelBump.states))])[0])>0:
			#print('Right wheel collided with', rightWheelBump.states[i].collision2_name.split('::')[0])
			collisionReward = -1
		elif len(chassisBump.states)>0:
			#print('Chassis Bumper collided')
			collisionReward = -1
		else:
			collisionReward = 0

		distance=np.sqrt((robotPosition[0]-goalPosition[0])**2+(robotPosition[1]-goalPosition[1])**2)
		distanceReward=-distance/np.sqrt(200)

		quaternion=(navibot_tf.transforms[0].transform.rotation.x,
					navibot_tf.transforms[0].transform.rotation.y,
					navibot_tf.transforms[0].transform.rotation.z,
					navibot_tf.transforms[0].transform.rotation.w)
		euler= tf.transformations.euler_from_quaternion(quaternion)
		euler=np.array(euler)
		#print(euler*180/np.pi)
		#print('Robot2World', euler[2]*180/np.pi)
		oppo=goalPosition[1]-robotPosition[1]
		adja=goalPosition[0]-robotPosition[0]
		if adja<0 and oppo<0:
			angle=-np.pi-np.arcsin(oppo/distance)
		elif adja<0 and oppo>0:
			angle=np.pi-np.arcsin(oppo/distance)
		else:
			angle=np.arcsin(oppo/distance)
		#print('Robot2Goal', angle*180/np.pi)
		#print('difference', (angle-euler[2])*180/np.pi)
		angleReward=-abs(angle-euler[2])/(4*np.pi)
		#print('angleReward', angleReward)

		goalReward=0
		if len([leftWheelBump.states[i].collision2_name.split('::')[0] == 'goal' for i in range(len(leftWheelBump.states))])>0:
			print('leftWheelBumber GOAL!')
			collisionReward = 0
			goalReward = 2
			self.isGoal=True
		elif len([rightWheelBump.states[i].collision2_name.split('::')[0] == 'goal' for i in range(len(rightWheelBump.states))])>0:
			print('rightWheelBump GOAL!')
			collisionReward = 0
			goalReward = 2
			self.isGoal=True
		elif len(np.where([chassisBump.states[i].collision2_name.split('::')[0] == 'goal' for i in range(len(chassisBump.states))])[0])>0:
			print('chassis GOAL!')
			collisionReward = 0
			goalReward = 2
			self.isGoal=True
		else:
			goalReward=0

		return collisionReward + distanceReward + goalReward + angleReward 

	def isFlipped(self):
		_,orientation=self.getRobotPosOri()
		if orientation[0]>0.5 or orientation[0]<-0.5 or \
		   orientation[1]>0.5 or orientation[1]<-0.5:
			return True
		else:
			return False

	def action(self, actionIdx):
		'''
		actionIdx   Action
					 L  R
		0			 0  1
		1			 1  0
		2			 1  1
		3			-1 -1
		'''

		vel=Twist()
		if actionIdx == 0:
			vel.angular.z =  1 * self.vel_curve
		elif actionIdx == 1:
			vel.angular.z = -1 * self.vel_curve
		elif actionIdx == 2:
			vel.linear.x =  1 * self.vel
		elif actionIdx == 3:
			vel.linear.x = -1 * self.vel
		else:
			raise ValueError('Action Index is not supported: ')
		
		self.velocityPub.publish(vel)

if __name__ == '__main__':
	dP=dataProcessor()
	

	while True:
		dP.action(0)
		pos,ori=dP.getRobotPosOri()
		rospy.loginfo(ori)