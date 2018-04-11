
'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np

import os
import sys
import time

    
import matplotlib.pyplot as plt
#%%

tf.reset_default_graph()
def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7) 


# train Parameters
seq_length = 30  #Y_seq_length + Range_Predict > seq_length
Range_Predict =29
Y_seq_length = 5
step_size = 10 # meter

data_dim = 4
hidden_dim = 10
output_dim = 1
learning_rate = 0.007
iterations = 401
LSTM_numLayers = 1
dropout_keep_probability = 0.1

save_every = 200
# Open, High, Low, Volume, Close

xy = np.loadtxt('experimentData.csv', delimiter=',')
VehiclePos_s = np.loadtxt('VehiclePos_s.csv', delimiter=',')
#%%
t=xy
xy = xy[::-1]  # reverse order (chronically ordered)
#%%
xy = MinMaxScaler(xy)


x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []


xc=x[:-( seq_length + Range_Predict)]
yc=y[:-( seq_length + Range_Predict)]
zero_ = np.array(0)
#%%
for i in range(0, len(yc) - seq_length-Range_Predict):
    _x  = xc[i:i + seq_length]
    
    _y = yc[i +Range_Predict: i + Y_seq_length + Range_Predict]
    
    #print(i, _x, '->', _y)
    dataX.append(_x)
    dataY.append(_y)


numpydataY=np.array(dataY)
numpydataY=numpydataY.reshape(len(dataY),Y_seq_length)

dataY_revised = numpydataY.tolist()


#%%
# train/test split
train_size = int(len(dataY_revised) * 0.7)
test_size = len(dataY_revised) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY_revised)])

#%%
    
    
# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, Y_seq_length,1])
#%%
# build a LSTM network
def lstm_cell():
    #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.nn.softsign)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.nn.softsign)
    #drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_probability)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(LSTM_numLayers)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)


#%%

Y_pred = tf.contrib.layers.fully_connected(outputs[:,-Y_seq_length:], output_dim, activation_fn=None)  # We use the last cell's output

#%%


# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y),name='loss')  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None,Y_seq_length, 1])
predictions = tf.placeholder(tf.float32, [None,Y_seq_length, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
#%%
losses = {'train':[],'test':[]}
RootMeanSquaredError = {'RMSE':[]}
#%%
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    file_writer=tf.summary.FileWriter('./logs2',sess.graph)

    # Training step
    for i in range(iterations):
        start=time.time()
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        losses['train'].append(step_loss)
        end=time.time()
        if (i %save_every ==0):
            save_path = saver.save(sess, "./tmp/model/i{}.ckpt".format(i))
            print("Model saved in file: ",save_path)
 
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * i/float(iterations)) + "% ... Training loss: " + str(step_loss)+ " ... Time{:.4f}: ".format(end-start))
        sys.stdout.flush()
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
            targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

#%%
testY_rev=testY.reshape(len(testY),Y_seq_length)

test_predict_rev = test_predict.reshape(len(testY),Y_seq_length)
Predicted_testY = test_predict_rev[:,(-Range_Predict-Y_seq_length+seq_length):]

testRange=np.array(list([range(len(testY)+Y_seq_length+Range_Predict)]))
Resized_testRange = testRange*step_size
plt.close('all') 
#%%
plt.figure(1)

plt.title('Measured velocity')
plt.ylabel('Velocity [km/h]')
plt.xlabel('Roadway position [m]')
plt.plot(testY_rev[:,0],color="red",lw=0.5)
plt.grid()
plt.show()
#%%
plt.figure(2)
#len(testY)
for i in range(0,len(testY)):
    plt.plot(Resized_testRange[0,i+seq_length:i+Range_Predict+Y_seq_length],Predicted_testY[i],'--')
plt.title('Estimation of velocity based on LSTM RNN')
plt.ylabel('Velocity [km/h]')
plt.xlabel('Roadway position [m]')
plt.grid()
plt.show()
#%%
plt.figure(3)
plt.plot(Resized_testRange[0,:-(Y_seq_length+Range_Predict)],testY_rev[:,0]*69.769,color="red",lw=0.5)

for i in range(0,len(testY)):
    plt.plot(Resized_testRange[0,i:i+Range_Predict+Y_seq_length-seq_length],Predicted_testY[i]*69.769,'-.',lw=0.5)
    #plt.plot(testRange[i:i+Y_seq_length],test_predict_rev[i],color="blue")
plt.title('Estimation of velocity based on LSTM RNN')
plt.ylabel('Velocity [km/h]')
plt.xlabel('Roadway position [m]')
plt.grid()
plt.show()

#%%

plt.figure(4)

plt.plot(losses['train'],color="red")
    #plt.plot(testRange[i:i+Y_seq_length],test_predict_rev[i],color="blue")
plt.title('Training loss per iteration')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.grid()
plt.show()



