from envControl import environmentControl
from dataProcessor import dataProcessor
from q_network import DeepQLearner
from data_set import DataSet
import time, os, pickle
import numpy as np
import sys
import rospy


#---------FILE-MANAGEMENT------------#    

def saveNetwork(dataFolder, numRounds, network):
    print('...Writing Network to: \n' + \
            dataFolder + '/NetFiles/' + 'Net_' + \
            str(numRounds) + '.pkl')
    netFile=open(dataFolder + '/NetFiles/' + 'Net_' + \
                str(numRounds) + '.pkl','wb')
    pickle.dump(network, netFile)
    netFile.close()
    print('Network written successfully.')

def loadNetwork(dataFolder, numRounds):
    #TODO: Add exceptions for files which are nonexistent
    print('...Reading Network from: \n' + \
        dataFolder + '/NetFiles/' + 'Net_' + \
        str(numRounds) + '.pkl')
    netFile=open(dataFolder + '/NetFiles/' + 'Net_'+ \
                str(numRounds) + '.pkl', 'rb')
    network=pickle.load(netFile)
    netFile.close()
    print('Network read succesfully.')
    return network

def saveDataSet(dataFolder, numRounds, dataSet):
    print('...Writing DataSet to: \n' + \
            dataFolder + '/DataSets/' + 'DataSet_' + \
            str(numRounds) + '.pkl')
    dataFile=open(dataFolder + '/DataSets/' + 'DataSet_' + \
                str(numRounds) + '.pkl','wb')
    pickle.dump(dataSet, dataFile)
    dataFile.close()
    print('DataSet written successfully.')

def loadDataSet(dataFolder, numRounds):
    #TODO: Add exceptions for files which are nonexistent
    print('...Reading DataSet from: \n' + \
        dataFolder + '/DataSets/' + 'DataSet_' + \
        str(numRounds) + '.pkl')
    dataFile=open(dataFolder + '/DataSets/' + 'DataSet_'+ \
                str(numRounds) + '.pkl', 'rb')
    dataSet=pickle.load(dataFile)
    dataFile.close()
    print('DataSet read succesfully.')
    return dataSet
    
def trainNetwork(dataSet, network, batchSize):
	batchStates, batchActions, batchRewards, batchTerminals= \
	            dataSet.randomBatch(batchSize)
	'''        
	print('batchStates: ', batchStates)
	print('BatchActions: ', batchActions)
	print('batchRewards: ', batchRewards)
	print('batchTerminals: ', batchTerminals)
	'''
	loss=network.train(batchStates, 
	                        batchActions, 
	                        batchRewards, 
	                        batchTerminals)
	return loss

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
    EPSILON_START= .55
    EPSILON_MIN= 0.1
    EPSILON_DECAY=0.05
    REPLAY_MEMORY_SIZE= 10000000
    RNG= np.random.RandomState()
    PHI_LENGTH=4
    STATE_SIZE=11
    ACTION_SIZE=4
    BATCH_SIZE=32
    DISCOUNT=0.99
    RHO=0.95 #RMS_DECAY
    MOMENTUM=-1
    LEARNING_RATE=0.0001
    RMS_EPSILON=0.01
    UPDATE_RULE='deepmind_rmsprop'
    BATCH_ACCUMULATOR='sum'
    LOAD_NET_NUMBER= 90000 #100000000 #50000000 
    SIZE_EPOCH=10000
    REPLAY_START_SIZE=100 #SIZE_EPOCH/2
    FREEZE_INTERVAL=5000


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
    SENSOR_RANGE_MAX=np.sqrt(800.)
    SENSOR_RANGE_MIN=0.
    VEL=0.5
    VEL_CURVE=0.2
    NUM_STEPS=5000

    UPDATE_TIME=0.5
    SPEED_UP=50 # 

def main(epsilon_start, load_net_number):
	sys.setrecursionlimit(2000)

	config=Configuration()
	config.EPSILON_START=epsilon_start
	config.LOAD_NET_NUMBER=load_net_number

	if config.LOAD_NET_NUMBER>0:
		dataSet=loadDataSet(config.DATA_FOLDER, config.LOAD_NET_NUMBER)
		network=loadNetwork(config.DATA_FOLDER, config.LOAD_NET_NUMBER)
		countTotalSteps = config.LOAD_NET_NUMBER
	else:
		network=DeepQLearner(config.STATE_SIZE,
	                        config.ACTION_SIZE,
	                        config.PHI_LENGTH,
	                        config.BATCH_SIZE,
	                        config.DISCOUNT,
	                        config.RHO,
	                        config.MOMENTUM,
	                        config.LEARNING_RATE,
	                        config.RMS_EPSILON,
	                        config.RNG,
	                        config.UPDATE_RULE,
	                        config.BATCH_ACCUMULATOR,
	                        config.FREEZE_INTERVAL)
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

	for i in range(4):
		state,reward=dP.getStateReward()
		action=np.random.randint(config.ACTION_SIZE)
		dP.action(action)
		dataSet.addSample(state,
						  action,
						  reward,
						  dP.isGoal)
		countTotalSteps+=1
		countSteps+=1
	try:
		while not quit:
			if countTotalSteps%1000==0:
				updateLearningFile(config.DATA_FOLDER, lossAverages, countTotalSteps)
				lossAverages=np.empty([0])
				print(countTotalSteps)

			eC.unpause()
			state,reward=dP.getStateReward()
			phi=dataSet.phi(state)
			#print('phi: ', phi)
			action=network.choose_action(phi, epsilon)
			#action=np.random.randint(config.ACTION_SIZE)
			#action=userAction()
			#time.sleep(0.5)
			dP.action(action)
			#print('state: ', state)
			#print('reward: ', reward)
			#print('action: ', action)
			
			# Check every 100 steps if is Flipped and Goal was reached
			if countSteps % 5 == 0:
				if dP.isGoal:
					print('The goal was reached after', countSteps, 'steps' )
					countSteps = 1
					eC.setRandomModelState(config.ROBOT_NAME)
					eC.setRandomModelState('goal')
					dP.isGoal=False
					

				if dP.isFlipped():
					eC.setRandomModelState(config.ROBOT_NAME)
					reward-=1
					print('Flipped!')
			
			reward-=0.01 # Reward that every step costs a little bit
			# After NUM_STEPS the chance is over
			if countSteps % config.NUM_STEPS == 0:
				countSteps = 1
				reward-=1
				eC.setRandomModelState(config.ROBOT_NAME)
				eC.setRandomModelState('goal')
				print('Your chance is over! Try again ...')

			eC.pause()
			dataSet.addSample(state,
							  action,
							  reward,
							  dP.isGoal)

			# Training
			if countTotalSteps>config.REPLAY_START_SIZE:
				loss=trainNetwork(dataSet, network, config.BATCH_SIZE)
				#print('Loss', loss)
	            # count How many trainings had been done
				batchCount+=1
	            # add loss to lossAverages
				lossAverages=np.append(lossAverages, loss)

			#Update Epsilon save dataSet, network
			if countTotalSteps % config.SIZE_EPOCH==0:
		        # Number of Epochs
				epochCount+=1
		      
		        # update Learning File
		        #updateLearningFile()
		      
		        # Update Epsilon
				if (epsilon - epsilonRate) < config.EPSILON_MIN:
					quit=True
				epsilon=max(epsilon - epsilonRate, config.EPSILON_MIN)        
				print('Epsilon updated to: ', epsilon)
				saveNetwork(config.DATA_FOLDER, countTotalSteps, network) 
				saveDataSet(config.DATA_FOLDER, countTotalSteps, dataSet)
				
			countTotalSteps+=1
			countSteps+=1

	except rospy.exceptions.ROSException:
		saveNetwork(config.DATA_FOLDER, countTotalSteps, network) 
		saveDataSet(config.DATA_FOLDER, countTotalSteps, dataSet)
		eC.close()
		main(epsilon, countTotalSteps)

#	except KeyboardInterrupt:
#		saveNetwork(config.DATA_FOLDER, countTotalSteps, network) 
#		saveDataSet(config.DATA_FOLDER, countTotalSteps, dataSet)
#		raise KeyboardInterrupt



if __name__ == '__main__':
	main(0.55, 90000)