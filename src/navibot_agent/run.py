from envControl import environmentControl
from dataProcessor import dataProcessor
from Agent import *
from data_set import DataSet
import time, os, pickle
import numpy as np
import sys
import rospy


#---------FILE-MANAGEMENT------------#    

def saveDataSet(dataFolder, numRounds, dataSet):
    print('...Writing DataSet to: \n' + \
            dataFolder + '/DataSets/' + 'DataSetTF_' + \
            str(numRounds) + '.pkl')
    dataFile=open(dataFolder + '/DataSets/' + 'DataSetTF_' + \
                str(numRounds) + '.pkl','wb')
    pickle.dump(dataSet, dataFile)
    dataFile.close()
    print('DataSet written successfully.')

def loadDataSet(dataFolder, numRounds):
    #TODO: Add exceptions for files which are nonexistent
    print('...Reading DataSet from: \n' + \
        dataFolder + '/DataSets/' + 'DataSetTF_' + \
        str(numRounds) + '.pkl')
    dataFile=open(dataFolder + '/DataSets/' + 'DataSetTF_'+ \
                str(numRounds) + '.pkl', 'rb')
    dataSet=pickle.load(dataFile)
    dataFile.close()
    print('DataSet read succesfully.')
    return dataSet

def userAction():
	while True:
		action = input()
		if action== 0 or action== 1 or action== 2 or action== 3 :
			return action
		else: 
			print('Invalid input Options: 1; 2; 3; 4 Try again: ')

def openLearningFile(dataFolder):
    learningFile=open(dataFolder + '/LearningFile/' + \
                        'learning.csv','w')
    learningFile.write('mean Loss, Epoch\n')
    learningFile.flush()
    learningFile.close()

def updateLearningFile(dataFolder, lossAverages, epochCount):
	learningFile=open(dataFolder + '/LearningFile/' + \
                        'learning.csv','a')
	out="{},{}\n".format(np.mean(lossAverages), epochCount)
	learningFile.write(out)
	learningFile.flush()
	learningFile.close()
"""
def openRewardFile():
    rewardFile=open(config.DATA_FOLDER + '/LearningFile/' + \
                        'reward.csv','w')
    rewardFile.write('Epoch, Reward\n')
    rewardFile.flush()

def updateRewardFile():
    out="{},{}\n".format(epochCount,
                         addUpReward)
    rewardFile.write(out)
	rewardFile.flush()
	"""


class Configuration:

    #------------------------
    # Agent/Network parameters:
    #------------------------
    EPSILON_START= 1.
    EPSILON_MIN= 0.1
    EPSILON_DECAY=0.05
    REPLAY_MEMORY_SIZE= 10000000
    RNG= np.random.RandomState()
    PHI_LENGTH=4
    STATE_SIZE=9
    ACTION_SIZE=4
    BATCH_SIZE=32
    LOAD_NET_NUMBER= 0
    SIZE_EPOCH=10000 #10000
    REPLAY_START_SIZE=100 #SIZE_EPOCH/2
    HIDDEN_LAYERS=[100, 100, 100]
    TAU = 0.001 # Porcentage that determines how much are parameters of mainQN net modified by targetQN
    GAMMA=0.99

	#------------------------
    # Environment Control:
    #------------------------
    home=os.path.expanduser("~")
    PATH_ROBOT = home + '/navibot_ws/src/navibot_description/urdf/navibot.xml'
    PATH_GOAL = home + '/navibot_ws/src/navibot_description/urdf/goal.xml'
    PATH_LAUNCHFILE = home + '/navibot_ws/src/navibot_gazebo/launch/navibot_world.launch'
    DATA_FOLDER = home + '/navibot_ws/src/navibot_agent'

	#------------------------
    # Data processor:
    #------------------------
    ROBOT_NAME='navibot'
    
    NUM_SENSOR_VAL=7
    SENSOR_RANGE_MAX=np.sqrt(200.)
    SENSOR_RANGE_MIN=0.
    VEL=0.5
    VEL_CURVE=0.2
    NUM_STEPS=1000

    UPDATE_TIME=0.5
    SPEED_UP=10 # 

def main():
	sys.setrecursionlimit(2000)

	config=Configuration()

	with open(config.DATA_FOLDER+'/config.txt', 'r') as f:
		configFile=f.read().split(',')

	print('Parameters', configFile)
	config.EPSILON_START=float(configFile[0])
	config.LOAD_NET_NUMBER=int(float(configFile[1]))


	agentTF=AgentTF(config.STATE_SIZE, 
					config.PHI_LENGTH, 
					config.ACTION_SIZE, 
					config.HIDDEN_LAYERS, 
					config.BATCH_SIZE,
					config.TAU,
					config.GAMMA)


	if config.LOAD_NET_NUMBER>0:
		dataSet=loadDataSet(config.DATA_FOLDER, config.LOAD_NET_NUMBER)
		agentTF.restore_model(config.DATA_FOLDER)
		countTotalSteps = config.LOAD_NET_NUMBER
	else:
	    # Initialize DataSet
		dataSet=DataSet(config.STATE_SIZE,
	                    config.REPLAY_MEMORY_SIZE,
	                    config.PHI_LENGTH,
	                    config.RNG)
		countTotalSteps = 0

		openLearningFile(config.DATA_FOLDER)

	eC=environmentControl(config.PATH_ROBOT, 
    					  config.PATH_GOAL,
    					  config.PATH_LAUNCHFILE)
	eC.spawn(config.ROBOT_NAME)
	eC.spawnGoal()
	eC.setRandomModelState(config.ROBOT_NAME)
	#eC.pause()

	dP=dataProcessor(eC, 
					 config.ROBOT_NAME,
					 config.PHI_LENGTH,
					 config.STATE_SIZE,
					 config.NUM_SENSOR_VAL,
					 config.SENSOR_RANGE_MAX,
					 config.SENSOR_RANGE_MIN,
					 config.VEL,
					 config.VEL_CURVE,
					 config.UPDATE_TIME,
					 config.SPEED_UP)


	lastState=np.zeros((1,config.STATE_SIZE))
	lastReward=0
	lastAction=0

	
	countSteps=0
	batchCount=0
	lossAverages=np.empty([0])
	epochCount=0

	epsilon=max(config.EPSILON_START, config.EPSILON_MIN)
	epsilonRate=config.EPSILON_DECAY

	quit=False

	try:
		for i in range(4):
			action=np.random.randint(config.ACTION_SIZE)
			dP.action(action)
			
			state,reward=dP.getStateReward()
			dataSet.addSample(lastState,
							  action,
							  reward,
							  state,
							  dP.isGoal)
			countTotalSteps+=1
			countSteps+=1
			lastState=state
		if config.EPSILON_START<0.09:
			quit=True
		while not quit:
			if countTotalSteps%1000==0:
				updateLearningFile(config.DATA_FOLDER, lossAverages, countTotalSteps)
				lossAverages=np.empty([0])
				print(countTotalSteps)


			phi=dataSet.phi(lastState)
			action=agentTF.getAction(phi, epsilon)
			#action=userAction()
			eC.unpause()
			dP.action(action)
			state,reward=dP.getStateReward()
			eC.pause()

			if dP.isGoal:
				print('The goal was reached in ', countSteps, ' steps')
				countSteps = 1
				eC.setRandomModelState(config.ROBOT_NAME)
				eC.setRandomModelState('goal')
				dP.isGoal=False
					
			if dP.flipped:
				eC.setRandomModelState(config.ROBOT_NAME)
				dP.flipped=False

			# After NUM_STEPS the chance is over
			if countSteps % config.NUM_STEPS == 0:
				countSteps = 1
				reward-=1
				eC.setRandomModelState(config.ROBOT_NAME)
				eC.setRandomModelState('goal')
				print('Your chance is over! Try again ...')

			#print(reward)

			dataSet.addSample(lastState,
							  action,
							  reward,
							  state,
							  dP.isGoal)
			
			# Training
			if countTotalSteps>config.REPLAY_START_SIZE and countTotalSteps%5==0:
				batchStates, batchActions, batchRewards, batchNextStates, batchTerminals= \
		            dataSet.randomBatch(config.BATCH_SIZE)
				loss = agentTF.train(batchStates, batchActions, batchRewards, batchNextStates, batchTerminals)
				#print('Loss', loss)
	            # count How many trainings had been done
				batchCount+=1
	            # add loss to lossAverages
				lossAverages=np.append(lossAverages, loss)

			
			#Update Epsilon save dataSet, network
			if countTotalSteps % config.SIZE_EPOCH==0:
		        # Number of Epochs
				epochCount+=1
		      
		        # Update Epsilon
				if (epsilon - epsilonRate) < config.EPSILON_MIN:
					quit=True
				epsilon=max(epsilon - epsilonRate, config.EPSILON_MIN)        
				print('Epsilon updated to: ', epsilon)
				
				agentTF.save_model( countTotalSteps, config.DATA_FOLDER)
				saveDataSet(config.DATA_FOLDER, countTotalSteps, dataSet)
			lastState=state
			countTotalSteps+=1
			countSteps+=1
	
	except rospy.exceptions.ROSException:
		agentTF.save_model( countTotalSteps, config.DATA_FOLDER)
		saveDataSet(config.DATA_FOLDER, countTotalSteps, dataSet)
		agentTF.close()
		eC.close()

		with open(config.DATA_FOLDER+'/config.txt', 'w') as f:
			out="{},{}".format(epsilon, countTotalSteps)
			f.write(out)

if __name__ == '__main__':
	main()