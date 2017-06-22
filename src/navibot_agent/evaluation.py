from Agent import *
from envControl import environmentControl
from dataProcessor import dataProcessor
from data_set import DataSet
import time, os, pickle
import numpy as np
import sys
import rospy
import matplotlib.pyplot as plt


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


EPSILON_START= 0.1
EPSILON_MIN= 0.1
EPSILON_DECAY=0.05
REPLAY_MEMORY_SIZE= 10000000
RNG= np.random.RandomState()
PHI_LENGTH=4
STATE_SIZE=9
ACTION_SIZE=4
BATCH_SIZE=32
LOAD_NET_NUMBER= 189377 #100000000 #50000000 
SIZE_EPOCH=5000 #10000
REPLAY_START_SIZE=100 #SIZE_EPOCH/2
HIDDEN_LAYERS=[512, 512, 512, 512, 512]
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

#------------------------
# Evaluation Parameters:
#------------------------

TRAINING_NEW_NETWORK=True
EVAL_OLD_NETWORK=False
STEPS_FOR_NUM_GAMES=True
NUM_GAMES=100

SKIP_STEP=5     # Training steps skipped in run.py 
TRAINING_ITER=70000#LOAD_NET_NUMBER/SKIP_STEP
AVERAGES_OVER_STEPS=1000 # Number of Steps over which 
ENVIRONMENT='Center Block'

sys.setrecursionlimit(2000)

dataSet=loadDataSet(DATA_FOLDER, LOAD_NET_NUMBER)

agentTF=AgentTF(STATE_SIZE, 
                PHI_LENGTH, 
                ACTION_SIZE, 
                HIDDEN_LAYERS, 
                BATCH_SIZE,
                TAU,
                GAMMA)


losses=np.empty([0])
meanLosses=np.empty([0])
iterations=np.empty([0])

if TRAINING_NEW_NETWORK:
    for i in range(1,TRAINING_ITER+1):
        if i % AVERAGES_OVER_STEPS==0:
            
            print('loss: ', np.mean(losses))

            iterations=np.append(iterations, i*SKIP_STEP)
            meanLosses=np.append(meanLosses, np.mean(losses))
            losses=np.empty([0])
            agentTF.save_model( i*SKIP_STEP, DATA_FOLDER+'/Evaluation')
        
        batchStates, batchActions, batchRewards, batchNextStates, batchTerminals= \
                        dataSet.randomBatch(BATCH_SIZE)
        loss = agentTF.train(batchStates, batchActions, batchRewards, batchNextStates, batchTerminals)
        losses=np.append(losses, loss)


    agentTF.save_model( i*SKIP_STEP, DATA_FOLDER+'/Evaluation')
    # Print Loss of this network

    fig1=plt.figure()
    plt.plot(iterations, meanLosses)
    plt.ylabel('Average Loss over ' + str(AVERAGES_OVER_STEPS) +'steps')
    plt.xlabel('Steps')
    plt.title('Loss Average ('+ str(ENVIRONMENT) + ' Environment)')
    plt.savefig(DATA_FOLDER + '/Evaluation/LossAverages_' + str(ENVIRONMENT) + '.png')

else : 
    agentTF.restore_model(DATA_FOLDER+'/Evaluation', 'model-5000.cptk')

if EVAL_OLD_NETWORK:
    graph=np.loadtxt(open('LearningFile/learning.csv','rb'), delimiter=',', skiprows=2)
    fig1=plt.figure()
    plt.plot(graph[:,1], graph[:,0])
    plt.ylabel('Average Loss over ' + str(AVERAGES_OVER_STEPS) +'steps')
    plt.xlabel('Steps')
    plt.title('Loss Average ('+ str(ENVIRONMENT) + ' Environment)')
    plt.savefig(DATA_FOLDER + '/Evaluation/LossAverages_' + str(ENVIRONMENT) + '.png')


if STEPS_FOR_NUM_GAMES:
    eC=environmentControl(PATH_ROBOT, 
                          PATH_GOAL,
                          PATH_LAUNCHFILE)
    eC.spawn(ROBOT_NAME)
    eC.spawnGoal()
    eC.setRandomModelState(ROBOT_NAME)
    #eC.pause()

    dP=dataProcessor(eC, 
                     ROBOT_NAME,
                     PHI_LENGTH,
                     STATE_SIZE,
                     NUM_SENSOR_VAL,
                     SENSOR_RANGE_MAX,
                     SENSOR_RANGE_MIN,
                     VEL,
                     VEL_CURVE,
                     UPDATE_TIME,
                     SPEED_UP)

    lastState=np.zeros((1,STATE_SIZE))
    
    countSteps=1
    steps_per_game=np.empty([0])
    num_games=0

    epsilon=max(EPSILON_START, EPSILON_MIN)
    epsilonRate=EPSILON_DECAY

    while num_games<NUM_GAMES:
        phi=dataSet.phi(lastState)
        action=agentTF.getAction(phi, epsilon)
        #action=userAction()
        eC.unpause()
        dP.action(action)
        state,reward=dP.getStateReward()
        eC.pause()

        if dP.isGoal:
            print('The goal was reached in ', countSteps, ' steps')
            steps_per_game=np.append(steps_per_game, countSteps)
            countSteps = 1
            num_games+=1
            eC.setRandomModelState(ROBOT_NAME)
            eC.setRandomModelState('goal')
            dP.isGoal=False
                
        if dP.flipped:
            eC.setRandomModelState(ROBOT_NAME)
            dP.flipped=False

        # After NUM_STEPS the chance is over
        if countSteps % NUM_STEPS == 0:
            steps_per_game=np.append(steps_per_game, countSteps)
            countSteps = 1
            num_games+=1
            reward-=1
            eC.setRandomModelState(ROBOT_NAME)
            eC.setRandomModelState('goal')
            print('Your chance is over! Try again ...')
        lastState=state
        countSteps+=1

    graph=np.loadtxt(open('LearningFile/learning.csv','rb'), delimiter=',', skiprows=2)
    fig1=plt.figure()
    plt.plot(np.arange(1, NUM_GAMES + 1), steps_per_game)
    plt.ylabel('Steps per Game')
    plt.xlabel('Game')
    plt.title('Steps per Game ('+ str(ENVIRONMENT) + ' Environment)')
    plt.savefig(DATA_FOLDER + '/Evaluation/StepsPerGame_' + str(ENVIRONMENT) + '.png')

    print('Mean:', np.mean(steps_per_game))

eC.close()
agentTF.close()