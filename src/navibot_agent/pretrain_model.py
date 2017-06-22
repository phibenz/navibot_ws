from Agent import *
from data_set import DataSet
import time, os, pickle
import numpy as np
import sys
import rospy



def loadDataSet(dataFolder, numRounds):
    #TODO: Add exceptions for files which are nonexistent
    print('...Reading DataSet from: \n' + \
        dataFolder + '/DataSets/' + 'DataSet_' + \
        str(numRounds) + '.pkl')
    dataFile=open(dataFolder + '/DataSets/' + 'DataSetTF_'+ \
                str(numRounds) + '.pkl', 'rb')
    dataSet=pickle.load(dataFile)
    dataFile.close()
    print('DataSet read succesfully.')
    return dataSet


EPSILON_START= 1.
EPSILON_MIN= 0.1
EPSILON_DECAY=0.05
REPLAY_MEMORY_SIZE= 10000000
RNG= np.random.RandomState()
PHI_LENGTH=4
STATE_SIZE=9
ACTION_SIZE=4
BATCH_SIZE=32
LOAD_NET_NUMBER= 0 #100000000 #50000000 
SIZE_EPOCH=5000 #10000
REPLAY_START_SIZE=100 #SIZE_EPOCH/2
HIDDEN_LAYERS=[256, 256, 256, 256]
TAU = 0.001 # Porcentage that determines how much are parameters of mainQN net modified by targetQN
GAMMA=0.99

#------------------------
# Environment Control:
#------------------------
home=os.path.expanduser("~")
DATA_FOLDER = home + '/navibot_ws/src/navibot_agent'

dataSet=loadDataSet(DATA_FOLDER, 51467)

agentTF=AgentTF(STATE_SIZE, 
                PHI_LENGTH, 
                ACTION_SIZE, 
                HIDDEN_LAYERS, 
                BATCH_SIZE,
                TAU,
                GAMMA)


lossAverages=np.empty([0])

for i in range(1000000):
    if i % 1000==0:
        print('loss: ', np.mean(lossAverages))
        lossAverages=np.empty([0])

    batchStates, batchActions, batchRewards, batchNextStates, batchTerminals= \
                    dataSet.randomBatch(BATCH_SIZE)
    loss = agentTF.train(batchStates, batchActions, batchRewards, batchNextStates, batchTerminals)
    lossAverages=np.append(lossAverages, loss)
agentTF.close()