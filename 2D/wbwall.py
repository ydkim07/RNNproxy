import os
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Import library we need
import tensorflow as tf
import numpy as np
		
# Specify initial seed to get consistent result
tf.set_random_seed(1)
#tf.random.set_random_seed(1)

# Arrange data
BHPtraining = np.loadtxt('BHPtraining.csv', delimiter=',', dtype = np.float32)
CumOiltraining = np.loadtxt('YWBW_all.csv', delimiter=',', dtype = np.float32)

# Parameters
n_neurons = 200
n_layers = 1
n_PRD = 9
n_INJ = 4
learning_rate = 0.001
n_iterations =800000 # training maximum iteration
trainingRatio = 0.5 # ratio of training set from example
trainTerminationLoss = 0.05
WaterPRDdummy = 20  # added to water PRD sample to handle loss calculation
#Wwi = 0 # water injection weight
#Wo = 1 # oil production weight
#Wwp = 0 # water production weight

# Calculated from user specified parameters
n_outputs = 2*n_PRD+n_INJ
n_steps = CumOiltraining.shape[1]//(2*n_PRD+n_INJ)
n_inputs = BHPtraining.shape[1]//n_steps
m = CumOiltraining.shape[0]

# Build a dataset
dataX = np.reshape(BHPtraining,(m, n_steps, n_inputs))
dataY = np.reshape(CumOiltraining,(m, n_steps, n_outputs))

# Prepare RNN cell 
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

# 1 hidden layer, n_neurons neurons per layer
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.LSTMCell(num_units = n_neurons, activation = tf.nn.relu), output_size = n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
#tf.contrib.rnn.GRUCell(num_units=n_neurons)
# Define Loss and Optimization                                                          
loss = tf.reduce_mean(tf.reduce_sum(tf.abs(outputs-Y),1)/tf.reduce_sum(Y,1))  # use Average as loss value, error calculation in Trehan
ter_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(outputs[:,1:,:]-Y[:,1:,:]),1)/tf.reduce_sum(Y[:,1:,:],1)) # unweighted loss
#ter_loss = Wwi*tf.reduce_mean(tf.reduce_sum(tf.abs(outputs[:,1:,0:n_INJ]-Y[:,1:,0:n_INJ]),1)/tf.reduce_sum(Y[:,1:,0:n_INJ],1)) + Wo*tf.reduce_mean(tf.reduce_sum(tf.abs(outputs[:,1:,n_INJ:n_INJ+n_PRD]-Y[:,1:,n_INJ:n_INJ+n_PRD]),1)/tf.reduce_sum(Y[:,1:,n_INJ:n_INJ+n_PRD],1)) + Wwp*tf.reduce_mean(tf.reduce_sum(tf.abs(outputs[:,1:,n_INJ+n_PRD:]-Y[:,1:,n_INJ+n_PRD:]),1)/tf.reduce_sum(Y[:,1:,n_INJ+n_PRD:],1))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_OP = optimizer.minimize(ter_loss)
init = tf.global_variables_initializer()

# Divide train/Dev
train_size = int(m * trainingRatio)
test_size = m - train_size
trainX, devX = np.array(dataX[0:train_size,:,:]), np.array(dataX[train_size:m,:,:])
trainY, devY = np.array(dataY[0:train_size,:,:]), np.array(dataY[train_size:m,:,:])
train_cost = []
dev_cost = []
train_ter_cost = []
dev_ter_cost = []

# Execution-w/o minibatch
# No minibatch since training size is small.
sess = tf.Session()
sess.run(init)

TerTrainMse = 50001 # dummy to start train
iteration = 0
while iteration < n_iterations and TerTrainMse > trainTerminationLoss:
  sess.run(training_OP, feed_dict = {X:trainX, Y:trainY})
  
  if iteration % 200 == 0:
    TrainMse = sess.run(loss, feed_dict = {X:trainX, Y:trainY})
    TerTrainMse = sess.run(ter_loss, feed_dict = {X:trainX, Y:trainY})
   # devMse = sess.run(loss, feed_dict={X:devX, Y:devY})
   # TerdevMse = sess.run(ter_loss, feed_dict={X:devX, Y:devY})
    print(iteration, "Train MSE", TrainMse)
   # print(iteration, "Dev  MSE", devMse)	
    print(iteration, "Train Ter MSE", TerTrainMse)
   # print(iteration, "Dev  Ter MSE", TerdevMse)
    train_cost.append(TrainMse)
    train_ter_cost.append(TerTrainMse)
  
  #dev_cost.append(devMse)
  
  #dev_ter_cost.append(TerdevMse)
  iteration = iteration + 1
 
# Save trained network on Result folder
saver = tf.train.Saver()
save_path = saver.save(sess, "Result/WBWall.ckpt")

# Save iteration vs cost
devMse = sess.run(loss, feed_dict={X:devX, Y:devY})
TerdevMse = sess.run(ter_loss, feed_dict={X:devX, Y:devY})
dev_cost.append(devMse)
dev_ter_cost.append(TerdevMse)
np.savetxt("./Result/trainCost.txt", train_cost)
np.savetxt("./Result/devCost.txt", dev_cost)
np.savetxt("./Result/trainTerCost.txt", train_ter_cost)
np.savetxt("./Result/devTerCost.txt", dev_ter_cost)

# Predict response using the trained model and save data to analyze using MATLAB.
Y_new = sess.run(outputs, feed_dict = {X:devX})
Y_new = Y_new.reshape(test_size,n_steps*(2*n_PRD+n_INJ))
np.savetxt("./Result/DevWBWall.csv", Y_new, delimiter =",")

# Training set prediction
Y_training_pred = sess.run(outputs, feed_dict = {X:trainX})
Y_training_pred = Y_training_pred.reshape(train_size,n_steps*(2*n_PRD+n_INJ))
np.savetxt("./Result/TrainingWBWall.csv", Y_training_pred, delimiter =",")

# Divide Y_new and Y_training_pred into separate water INJ, oil PRD, and water PRD files and save
waterINJ_new = np.zeros((test_size, n_steps*n_INJ))
oilPRD_new = np.zeros((test_size, n_steps*n_PRD))
waterPRD_new = np.zeros((test_size, n_steps*n_PRD))
waterINJ_training = np.zeros((train_size, n_steps*n_INJ))
oilPRD_training = np.zeros((train_size, n_steps*n_PRD))
waterPRD_training = np.zeros((train_size, n_steps*n_PRD))

for time in range(n_steps):
	waterINJ_new[:,n_INJ*time:n_INJ*time+n_INJ] = np.array(Y_new[:,(2*n_PRD+n_INJ)*time:(2*n_PRD+n_INJ)*time+n_INJ])
	oilPRD_new[:,n_PRD*time:n_PRD*time+n_PRD] = np.maximum(0,np.array(Y_new[:,(2*n_PRD+n_INJ)*time+n_INJ:(2*n_PRD+n_INJ)*time+n_INJ+n_PRD]))
	waterPRD_new[:,n_PRD*time:n_PRD*time+n_PRD] = np.maximum(WaterPRDdummy,np.array(Y_new[:,(2*n_PRD+n_INJ)*time+n_INJ+n_PRD:(2*n_PRD+n_INJ)*time+n_INJ+n_PRD+n_PRD]))
	waterINJ_training[:,n_INJ*time:n_INJ*time+n_INJ] = np.array(Y_training_pred[:,(2*n_PRD+n_INJ)*time:(2*n_PRD+n_INJ)*time+n_INJ])
	oilPRD_training[:,n_PRD*time:n_PRD*time+n_PRD] = np.maximum(0,np.array(Y_training_pred[:,(2*n_PRD+n_INJ)*time+n_INJ:(2*n_PRD+n_INJ)*time++n_INJ+n_PRD]))
	waterPRD_training[:,n_PRD*time:n_PRD*time+n_PRD] = np.maximum(WaterPRDdummy,np.array(Y_training_pred[:,(2*n_PRD+n_INJ)*time+n_INJ+n_PRD:(2*n_PRD+n_INJ)*time++n_INJ+n_PRD+n_PRD]))

waterPRD_new = waterPRD_new - WaterPRDdummy # subtract dummy
waterPRD_training = waterPRD_training - WaterPRDdummy

np.savetxt("./Result/DevWBWoil.csv", oilPRD_new, delimiter = ",")
np.savetxt("./Result/DevWBWwaterINJ.csv", waterINJ_new, delimiter = ",")
np.savetxt("./Result/DevWBWwaterPRD.csv", waterPRD_new, delimiter = ",")
np.savetxt("./Result/TrainingWBWoil.csv", oilPRD_training, delimiter = ",")
np.savetxt("./Result/TrainingWBWwaterINJ.csv", waterINJ_training, delimiter = ",")
np.savetxt("./Result/TrainingWBWwaterPRD.csv", waterPRD_training, delimiter = ",")

#Close session
sess.close()

