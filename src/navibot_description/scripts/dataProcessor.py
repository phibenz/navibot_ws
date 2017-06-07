import rospy
import time
import numpy as np

from envControl import environmentControl

from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates, ContactsState
from geometry_msgs.msg import Twist

from std_srvs.srv import Empty


class dataProcessor:
	def __init__(self, environmentController, robotName, updatesPerStep, phiLength, stateSize, numSensorVal, sensorRangeMax, sensorRangeMin, vel, vel_curve):

		rospy.init_node('dataProcessor', anonymous=True)

		self.envC=environmentController

		self.robotName = robotName
		self.robotIndex, self.goalIndex = self.getIndeces()
		self.updatesPerStep=updatesPerStep 
		self.phiLength=phiLength
		self.deltaStep=int(self.updatesPerStep/(self.phiLength-1))
		self.isGoal=False

		self.stateSize=stateSize # TODO maybe as input
		self.numSensorVal=numSensorVal
		self.SenorRangeMax=sensorRangeMax #TODO: Needs clarification
		self.SensorRangeMin=sensorRangeMin

		self.vel=vel
		self.vel_curve=vel_curve
		#self.leftWheelVelPub = rospy.Publisher("/navibot/left_wheel_hinge_velocity_controller/command", 
		#										Float64, queue_size=10)
		#self.rightWheelVelPub = rospy.Publisher("/navibot/right_wheel_hinge_velocity_controller/command", 
		#										Float64, queue_size=10)
		self.velocityPub = rospy.Publisher("/navibot/velocity", Twist, queue_size=10)

	def getIndeces(self):
		# Helper function
		modelNames=rospy.wait_for_message("/gazebo/model_states", ModelStates).name
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

	def getStateReward(self):
		#self.envC.unpause()
		time=rospy.wait_for_message("/clock", Clock).clock
		lastTime=time
		counter=0
		reward=0
		state=np.zeros((1, self.stateSize))
		while(counter<=self.updatesPerStep):
			time=rospy.wait_for_message("/clock", Clock).clock
			if lastTime<time:
				if counter==self.updatesPerStep:
					laserData=np.array(rospy.wait_for_message("/navibot/laser/scan", LaserScan).ranges)
					laserData[np.where(np.isinf(laserData))[0]]=0.
					state[0,0:self.numSensorVal]=(laserData-self.SensorRangeMin)/(self.SenorRangeMax-self.SensorRangeMin)
					
					robotPosition,robotOrientation=self.getRobotPosOri()
					goalPosition=self.getGoalPos()
					state[0,7]=(np.sqrt((robotPosition[0]-goalPosition[0])**2+(robotPosition[1]-goalPosition[1])**2))/np.sqrt(800)
					opposite=(goalPosition[0]-robotPosition[0])
					adjacent=(goalPosition[1]-robotPosition[1])
					hypotenuse=np.sqrt((robotPosition[0]-goalPosition[0])**2+(robotPosition[1]-goalPosition[1])**2)
					phi=np.arcsin(opposite/hypotenuse)
					if adjacent< 0 and opposite < 0:
						phi-=np.pi/2
					elif adjacent < 0 and opposite > 0:
						phi+=np.pi/2
					state[0,8]=phi/np.pi
					reward=self.getReward(robotPosition,robotOrientation,goalPosition)
				counter+=1
				lastTime=time
		#self.envC.pause()
		return state, reward

	def getRobotPosOri(self):
		#wasPaused=False
		#if self.isPaused():
		#	self.envC.unpause()
		#	wasPaused=True
		roboPO=rospy.wait_for_message("/gazebo/model_states", ModelStates).pose[self.robotIndex]

		robotPosition=np.array((roboPO.position.x, roboPO.position.y, roboPO.position.z))
		robotOrientation=np.array((roboPO.orientation.x, roboPO.orientation.y, roboPO.orientation.z))
		#if wasPaused:
		#	self.envC.pause()
		return robotPosition, robotOrientation

	def getGoalPos(self):

		goalPose=rospy.wait_for_message("/gazebo/model_states", ModelStates).pose[self.goalIndex]
		goalPos=np.array((goalPose.position.x, goalPose.position.y, goalPose.position.z))
		
		return goalPos

	def isPaused(self):
		return self.envC.physicsProp_client.call().pause

	def getReward(self, robotPosition, robotOrientation, goalPosition):
		leftWheelBump=rospy.wait_for_message("/navibot/left_wheel_bumper", ContactsState)
		rightWheelBump=rospy.wait_for_message("/navibot/right_wheel_bumper", ContactsState)
		chassisBump=rospy.wait_for_message("/navibot/chassis_bumper", ContactsState)
		
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
		distanceReward=-distance/np.sqrt(800) # Noramlize. Longest distance is the diagonal over the 20x20 field

		goalReward=0
		if len([leftWheelBump.states[i].collision2_name.split('::')[0] == 'goal' for i in range(len(leftWheelBump.states))])>0:
			print('leftWheelBumber GOAL!')
			goalReward = 1
			self.isGoal=True
		elif len([rightWheelBump.states[i].collision2_name.split('::')[0] == 'goal' for i in range(len(rightWheelBump.states))])>0:
			print('rightWheelBump GOAL!')
			goalReward = 1
			self.isGoal=True
		elif len(np.where([chassisBump.states[i].collision2_name.split('::')[0] == 'goal' for i in range(len(chassisBump.states))])[0])>0:
			print('chassis GOAL!')
			goalReward = 1
			self.isGoal=True
		else:
			goalReward=0

		return collisionReward + distanceReward + goalReward

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