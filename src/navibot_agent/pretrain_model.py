from q_network import DeepQLearner
from data_set import DataSet
import time, os, pickle
import numpy as np

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

def saveNetwork(dataFolder, numRounds, network):
    print('...Writing Network to: \n' + \
            dataFolder + '/NetFiles/' + 'Net_' + \
            str(numRounds) + '.pkl')
    netFile=open(dataFolder + '/NetFiles/' + 'Net_' + \
                str(numRounds) + '.pkl','wb')
    pickle.dump(network, netFile)
    netFile.close()
    print('Network written successfully.')


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


# Params

#------------------------
# Agent/Network parameters:
#------------------------
EPSILON_START= .95
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
LEARNING_RATE=0.000001
RMS_EPSILON=0.01
UPDATE_RULE='deepmind_rmsprop'
BATCH_ACCUMULATOR='sum'
LOAD_NET_NUMBER= 10000 #100000000 #50000000 
SIZE_EPOCH=10000
REPLAY_START_SIZE=100 #SIZE_EPOCH/2
FREEZE_INTERVAL=5000

LOAD_NET_NUMBER = 90000
home=os.path.expanduser("~")
DATA_FOLDER = home + '/navibot_ws/src/navibot_agent'

dataSet=loadDataSet(DATA_FOLDER, LOAD_NET_NUMBER)
network=DeepQLearner(STATE_SIZE,
	                        ACTION_SIZE,
	                        PHI_LENGTH,
	                        BATCH_SIZE,
	                        DISCOUNT,
	                        RHO,
	                        MOMENTUM,
	                        LEARNING_RATE,
	                        RMS_EPSILON,
	                        RNG,
	                        UPDATE_RULE,
	                        BATCH_ACCUMULATOR,
	                        FREEZE_INTERVAL)


lossAverages=np.empty([0])

for i in range(90000):

	if i % 1000==0:
		print('loss: ', np.mean(lossAverages))
		lossAverages=np.empty([0])

	loss=trainNetwork(dataSet, network, BATCH_SIZE)
			#print('Loss', loss)
            # count How many trainings had been done
    # add loss to lossAverages
	lossAverages=np.append(lossAverages, loss)

# saveNetwork(DATA_FOLDER, 111, network)