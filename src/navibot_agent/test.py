import tensorflow as tf
import numpy as np
from AgentTF import *
from data_setTF import DataSet

dataSet=DataSet(11,10000,4, np.random.RandomState())
Agent=AgentTF(11,4,4,[256,256,256], 32, 0.99, 0.01)

for i in range(100):
	state=np.random.rand()*np.arange(11)
	action=np.random.randint(0,4)
	reward=np.random.rand()
	nextState=np.random.rand()*np.arange(11)
	terminal=i%3==0

	dataSet.addSample( state, action, reward, nextState, terminal)

phi=dataSet.phi(state)
print('phi',phi)
action = Agent.getAction(phi, 0)
print(action)

#print dataSet.lastPhi()
states, actions, rewards, nextStates, terminals = dataSet.randomBatch(32)
#print np.shape(states)
#print (actions)
#print np.shape(actions)
#print np.shape(rewards)
#print np.shape(nextStates)
#print np.shape(terminals)

loss=Agent.train(states, actions, rewards, nextStates, terminals)
print(loss)
#myBatch=np.array(([state, action, reward, nextState, terminal]))
#print myBatch

Agent.close()