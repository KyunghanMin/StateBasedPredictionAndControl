# -*- coding: utf-8 -*-
"""
Data: 2018.03.20
Author: Kyunghan Min (kyunghah.min@gmail.com)
Description: Prediction model for stop tendency of vehicle
"""
#%% Import library
import numpy as np
import math
import scipy.io
import random
import sys 
import matplotlib.pyplot as plt
import shelve
import tensorflow as tf
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K

#%% Initialization Keras session
K.clear_session()

#%% Declare local function
def NormColArry(data):
    ''' Normalization of data array
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    '''
    tmpLoc_Num_NonZeroCriticValue = 1e-7;
    tmpLoc_Arry_numerator = data - np.min(data, 0);
    tmpLoc_Num_denominator = np.max(data, 0) - np.min(data, 0);
    # noise term prevents the zero division
    MaximumValue = np.max(data,0);
    MaximumValue_LastCol = MaximumValue[-1];
    return tmpLoc_Arry_numerator / (tmpLoc_Num_denominator + tmpLoc_Num_NonZeroCriticValue), MaximumValue_LastCol;
#%% Import data and normalization
DataLoad = scipy.io.loadmat('TestUpData.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];
#%% Model configuration
ModelConfig_NumInputSequence = 10;
ModelConfig_NumFeature = 3;
ModelConfig_NumLstmUnit = 100;
ModelConfig_NumOutputSize = 1;
ModelConfig_NumEpoch = 2;
ModelConfig_NumBatch = 100;
#%% Data arrange
DataNum = 0;
DataSetNum = 1;
# Data set arrange
DataSet_X = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumInputSequence,ModelConfig_NumFeature));
DataSet_Y = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumOutputSize));
# Slicing and cooking
for key in DataLoad:
    DataCurrent = DataLoad[key];
    DataSize = DataCurrent.shape[0];
    DataSetNum = DataSetNum+1;
    DataSetNumCurr = 1;
    for index in range(0,DataSize-10):        
        DataSet_X[DataNum,:,:] = DataCurrent[index:index+ModelConfig_NumInputSequence,0:-1]
        DataSet_Y[DataNum,:] = DataCurrent[index+ModelConfig_NumInputSequence,0];
        DataNum = DataNum+1;
        DataSetNumCurr = DataSetNumCurr + 1;
#%% Design model
# Model structure - Reccurent neural network
# Input layer - LSTM (Output size = 100, Input size = (50,3))
#             - 5 second sequential data
#             - Inputs: Acceleration, Velocity, Distance
# Hidden layer - LSTM (Output size = 100)
# Output layer - Softmax activation (Output size = 3)
model = Sequential()
model.add(LSTM(ModelConfig_NumLstmUnit, return_sequences=True,input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumFeature)))
model.add(LSTM(ModelConfig_NumLstmUnit))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')
#%% Training data set
TrainConfig_TrainRatio = 0.8;
TrainConfig_ValidRatio = 1 - TrainConfig_TrainRatio;
TrainConfig_NumTrainSet = int(DataNum*TrainConfig_TrainRatio);
TrainConfig_DataSetList = list(range(DataNum))
TrainConfig_TrainSetList = TrainConfig_DataSetList[0:TrainConfig_NumTrainSet];
TrainConfig_ValidSetList = TrainConfig_DataSetList[TrainConfig_NumTrainSet:];

x_train = DataSet_X[TrainConfig_TrainSetList,:,:];
y_train = DataSet_Y[TrainConfig_TrainSetList,:];

x_valid = DataSet_X[TrainConfig_ValidSetList,:,:];
y_valid = DataSet_Y[TrainConfig_ValidSetList,:];
#%% Fit network

model.fit(x_train, y_train,
          batch_size=ModelConfig_NumBatch, epochs=5, shuffle=True,
          validation_data=(x_valid, y_valid))    
    