'''Code for Q-Learning as described in:
Playing Atari with Deep Reinforcement Learning
'''

import numpy
import random

import theano
import lasagne
import theano.tensor as T
from updates import deepmind_rmsprop

# for testing purposes
# from data_set import DataSet

class DeepQLearner:
    '''
    Deep Q-Learning network using lasagne
    '''

    def __init__(self, stateSize, actionSize, numFrames, batchSize,
                discount, rho, momentum, learningRate, rmsEpsilon, 
                rng, updateRule, batchAccumulator, freezeInterval):
        self.stateSize=stateSize
        self.actionSize=actionSize
        self.numFrames=numFrames
        self.batchSize=batchSize
        self.discount=discount
        self.rho=rho
        self.momentum=momentum
        self.learningRate=learningRate
        self.rmsEpsilon=rmsEpsilon
        self.rng=rng
        self.updateRule=updateRule
        self.batchAccumulator=batchAccumulator
        self.freezeInterval=freezeInterval

        lasagne.random.set_rng(self.rng)

        self.updateCounter=0

        self.lOut=self.buildNetwork(self.stateSize, self.actionSize, 
                               self.numFrames,self.batchSize)

        if self.freezeInterval>0:
            self.nextLOut=self.buildNetwork(self.stateSize, self.actionSize, 
                                        self.numFrames,self.batchSize)
            self.resetQHat()


        states=T.ftensor3('states')
        nextStates=T.ftensor3('nextStates')
        rewards=T.fcol('rewards')
        actions=T.icol('actions')
        terminals=T.icol('terminals')

        # Shared variables for teaching from a minibatch of replayed 
        # state transitions, each consisting of num_frames + 1 (due to
        # overlap) states, along with the chosen action and resulting 
        # reward and termninal status.
        self.states_shared=theano.shared(
                                numpy.zeros((self.batchSize, self.numFrames+1, self.stateSize),
                                dtype=theano.config.floatX))
        self.rewards_shared=theano.shared(
                                numpy.zeros((self.batchSize,1), 
                                dtype=theano.config.floatX),
                                broadcastable=(False, True))
        self.actions_shared=theano.shared(
                                numpy.zeros((self.batchSize,1),
                                dtype='int32'),
                                broadcastable=(False, True))
        self.terminals_shared=theano.shared(
                                numpy.zeros((self.batchSize,1),
                                dtype='int32'),
                                broadcastable=(False,True))

        # Shared variable for a single state, to calculate qVals
        self.state_shared=theano.shared(
                                numpy.zeros((self.numFrames, self.stateSize),
                                dtype=theano.config.floatX))

        qVals=lasagne.layers.get_output(self.lOut, states)
        
        if self.freezeInterval>0:
            nextQVals=lasagne.layers.get_output(self.nextLOut, nextStates)
        else:
            nextQVals=lasagne.layers.get_output(self.lOut, nextStates)
            nextQVals=theano.gradient.disconnected_grad(nextQVals)
        
        # Cast terminals to floatX
        terminalsX=terminals.astype(theano.config.floatX)
        # T.eq(a,b) returns a variable representing the nogical
        # EQuality (a==b)
        actionmask=T.eq(T.arange(self.actionSize).reshape((1,-1)),
                        actions.reshape((-1,1))).astype(theano.config.floatX)
        
        target=(rewards + (T.ones_like(terminalsX)-terminalsX) *
                self.discount*T.max(nextQVals, axis=1, keepdims=True))
        output=(qVals*actionmask).sum(axis=1).reshape((-1,1))
        diff=target-output

        # no if clip delta, since clip-delta=0
        
        loss=(diff**2)

        if self.batchAccumulator=='sum':
            loss=T.sum(loss)
        elif self.batchAccumulator=='mean':
            loss=T.mean(loss)
        else:
            raise ValueError('Bad accumulator: {}'.format(batch_accumulator))
        
        params=lasagne.layers.helper.get_all_params(self.lOut)
        train_givens={
            states: self.states_shared[:,:-1],
            nextStates: self.states_shared[:,1:],
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
            
        if self.updateRule=='rmsprop':
            updates=lasagne.updates.rmsprop(loss, params, self.learningRate,
                                             self.rho, self.rmsEpsilon)

        elif self.updateRule=='deepmind_rmsprop':
            updates=deepmind_rmsprop(loss, params, self.learningRate, 
                                            self.rho, self.rmsEpsilon)
        else:
            raise ValueError('Unrecognized update: {}'. format(updateRule))
        
        if self.momentum>0:
            updates = lasagne.updates.apply_momentum(updates, None, self.momentum)
        
        self._train=theano.function([], [loss], updates=updates,
                                    givens=train_givens)
        q_givens={
            states: self.state_shared.reshape((1,
                                            self.numFrames,
                                            self.stateSize))
        }
        
        # self._q_vals=theano.function([],qVals[0], givens=q_givens)
        self._q_vals=theano.function([], qVals[0], givens=q_givens)
    
    def train(self, states, actions, rewards, terminals):
        """
        Train one batch
        Args:
        states - b x (f+1) x sS numpy array,
                b: batch size
                f: number of frames
                sS: stateSize
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self.states_shared.set_value(states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if(self.freezeInterval>0 and
            self.updateCounter % self.freezeInterval==0):
            self.resetQHat()
        
        loss=self._train()
        
        self.updateCounter+=1
        return loss

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()
    
    def choose_action(self, state, epsilon):
        if self.rng.rand()<epsilon:
            return self.rng.randint(0,self.actionSize)
        q_vals=self.q_vals(state)
        return numpy.argmax(q_vals)

    def resetQHat(self):
        allParams=lasagne.layers.helper.get_all_param_values(self.lOut)
        lasagne.layers.helper.set_all_param_values(self.nextLOut, allParams)


    def buildNetwork(self, inputDim, outputDim, numFrames, batchSize):
        lIn=lasagne.layers.InputLayer(
            shape=(None, numFrames, inputDim)
        )
        
        lHidden1=lasagne.layers.DenseLayer(
                    lIn,
                    num_units=100,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.Normal(.01),
                    b=lasagne.init.Constant(0.1)
        )
        
        lHidden2=lasagne.layers.DenseLayer(
                    lHidden1,
                    num_units=100,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.Normal(.01),
                    b=lasagne.init.Constant(.1)
        )

        lHidden3=lasagne.layers.DenseLayer(
                    lHidden2,
                    num_units=100,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.Normal(.01),
                    b=lasagne.init.Constant(.1)
        )

        lOut=lasagne.layers.DenseLayer(
                    lHidden3,
                    num_units=outputDim,
                    nonlinearity=None,
                    W=lasagne.init.Normal(.01),
                    b=lasagne.init.Constant(.1)
        )
        return lOut

#--------------Test------------------
    def trainTest(self):
        dataset=DataSet(3, 10, 4, numpy.random.RandomState())
        for i in range(10):
            state = numpy.random.random(3)*480
            action = numpy.random.randint(3)
            reward = numpy.random.random()
            terminal = False
            dataset.addSample(state, action, reward, terminal)
        S, A, R, T = dataset.randomBatch(8)
        print('S: ')
        print(S)
        print('A: ')
        print(A)
        print('R: ')
        print(R)
        print('T: ')
        print(T)
        loss=self.train(S,A,R,T)
        print('Loss: ', loss)
     
def main():
    net=DeepQLearner(3, 3, 4, 8, 0.995, 0.99,0, 0.0005, 1e-6,
    numpy.random.RandomState(),'rmsprop', 'sum', 10000)
    net.trainTest()
       
if __name__ == '__main__':
    main()

