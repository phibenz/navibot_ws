import numpy as np
import tensorflow as tf

class AgentTF:
	def __init__(state_size, phi_length, action_size, hidden_layers):
		
		self.state_size=state_size
		self.phi_length=phi_length
		self.action_size=action_size
		self.hidden_layers=hidden_layers

		# tensorflow
		tf.reset_default_graph()

		self.mainQN = DQN("mainQN", self.state_size*self.phi_length, self.action_size, 
								self.hidden_layers, self.phi_length)
	    self.targetQN = DQN("targetQN", self.state_size*self.phi_length, self.action_size,
	    						self.hidden_layers, self.phi_length)

	    self.trainables = tf.trainable_variables()
	    self.target_ops = self.set_target_graph_vars(trainables, tau)

	    #START
	    init = tf.global_variables_initializer()
	    self.sess = tf.Session()
	    self.sess.run(init)
	    # Set the target network to be equal to the primary network
	    self.update_target_graph(self.target_ops, self.sess)

	""" Auxiliary Methods """
	# Originally called updateTargetGraph
	def set_target_graph_vars(self, tfVars, tau):
	    total_vars = len(tfVars)
	    op_holder = []

	    for idx,var in enumerate(tfVars[0:total_vars//2]): # Select the first half of the variables (mainQ net)
	        op_holder.append( tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))

	    return op_holder
	# Originally called updateTarget
	def update_target_graph(self, op_holder):
	    for op in op_holder:
	        self.sess.run(op)

    def getAction(self, state, epsilon):
        # State has to be np.array(44, 1)
        if np.size(state) != (44,1):
            raise ValueError
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.input:[state]})[0]

    def train(self, train_batch):
    	# Train_batch = [s,a,r,s1,d]
    	#Perform the Double-DQN update to the target Q-values
		Q1 = self.sess.run(self.mainQN.predict,
		              feed_dict={self.mainQN.input:np.vstack(train_batch[:,3])})

		Q2 = self.sess.run(self.targetQN.Qout,
		              feed_dict={self.targetQN.input:np.vstack(train_batch[:,3])})

		end_multiplier = -(train_batch[:,4] - 1)
		doubleQ = Q2[range(self.batch_size),Q1]
		targetQ = train_batch[:,2] + (self.gamma*doubleQ*end_multiplier)

		# Update the network with our target values.
		_ = sess.run(self.mainQN.updateModel,
		             feed_dict={self.mainQN.input:np.vstack(train_batch[:,0]),
		             self.mainQN.targetQ:targetQ,
		             self.mainQN.actions:train_batch[:,1]})

		# Set the target network to be equal to the primary
		self.update_target_graph(self.target_ops, self.sess)


class DQN():
    def __init__(self, net_name, state_size, action_size, hiddens, num_frames):
        self.action_size=action_size
        self.net_name = net_name


        with tf.variable_scope(self.net_name):

            self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            #self.input_state = tf.reshape(self.state, [-1, num_frames * state_size])

            # Weights of each layer
            self.W = {
                'W1': self.init_weight("W1", [state_size, hiddens[0]]),
                'W2': self.init_weight("W2", [hiddens[0], hiddens[1]]),
                'W3': self.init_weight("W3", [hiddens[1], hiddens[2]]),
                'AW': self.init_weight("AW", [hiddens[2]//2, action_size]),
                'VM': self.init_weight("VM", [hiddens[2]//2, 1])
            }

            self.b = {
                'b1': self.init_bias("b1", hiddens[0]),
                'b2': self.init_bias("b2", hiddens[1]),
                'b3': self.init_bias("b3", hiddens[2])
            }

            # Layers
            self.hidden1 = tf.nn.relu(tf.add(tf.matmul(self.input, self.W['W1']), self.b['b1']))
            self.hidden2 = tf.nn.relu(tf.add(tf.matmul(self.hidden1, self.W['W2']), self.b['b2']))
            self.hidden3 = tf.nn.relu(tf.add(tf.matmul(self.hidden2, self.W['W3']), self.b['b3']))

            # Compute the Advantage, Value, and total Q value
            self.A, self.V = tf.split(self.hidden3, 2, 1)
            self.Advantage = tf.matmul(self.A, self.W['AW'])
            self.Value = tf.matmul(self.V, self.W['VM'])
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

            # Calcultate the action with highest Q value
            self.predict = tf.argmax(self.Qout, 1)

            # Compute the loss (sum of squared differences)
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_one_hot), axis=1)
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)

            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)

    def init_weight(self, name, shape):
        return tf.get_variable(name=name, shape=shape, initializer=initializer.xavier_initializer(), dtype=tf.float32)

    def init_bias(self, name, shape):
        return tf.Variable(tf.random_normal([shape]))
        #initializer = tf.constant(np.random.rand(shape))
        #return tf.get_variable(name=name, initializer=initializer, dtype=tf.float32)