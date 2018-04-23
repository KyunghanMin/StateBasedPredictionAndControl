# -*- coding: utf-8 -*-
"""
Data: 2018.03.20
Author: Kyunghan Min (kyunghah.min@gmail.com)
Description: Prediction model for stop tendency of vehicle
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:43:53 2018

@author: Kyunghan (kyunghah.min@gmail.com)

Description: Prediction model for stop tendency of vehicle
"""
#%% Import library
import os
import numpy as np
import math
import scipy.io
import random
import sys 
import matplotlib.pyplot as plt
import shelve
import tensorflow as tf
from pathlib import Path
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Reshape
from keras.layers import SimpleRNN
from keras import backend as K
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
    MaxVal = np.max(np.max(data,0),0);
    MinVal = np.min(np.min(data,0),0);
    tmpLoc_Num_NonZeroCriticValue = 1e-7;
    tmpLoc_Arry_numerator = data - MinVal;
    tmpLoc_Num_denominator = MaxVal - MinVal;    
    return tmpLoc_Arry_numerator / (tmpLoc_Num_denominator + tmpLoc_Num_NonZeroCriticValue), MaxVal, MinVal;
#%% Import data and normalization
cdir = os.getcwd()
data_dir = os.chdir('../data')

DataLoad = scipy.io.loadmat('TestUpData.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];

os.chdir(cdir)
#%% Model configuration
ModelConfig_NumInputSequence = 10;
ModelConfig_NumFeature = 4;
ModelConfig_NumLstmUnit = 10;
ModelConfig_NumOutputSize = 1;
ModelConfig_NumEpoch = 2;
ModelConfig_NumBatch = 50;
#%% Data arrange
DataNum = 0;
DataSetNum = 1;
# Data set arrange
DataSet_X = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumInputSequence,ModelConfig_NumFeature));
DataSet_X_Buff = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumInputSequence,ModelConfig_NumFeature));
DataSet_Y = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumOutputSize));

# Slicing and cooking
for key in DataLoad:
    DataCurrent = DataLoad[key];
    DataSize = DataCurrent.shape[0];
    DataSetNum = DataSetNum+1;
    DataSetNumCurr = 1;
    for index in range(0,DataSize-10):        
        DataSet_X[DataNum,:,:] = DataCurrent[index:index+ModelConfig_NumInputSequence,1:-1]
        DataSet_Y[DataNum,:] = DataCurrent[index+ModelConfig_NumInputSequence,0];
        DataNum = DataNum+1;
        DataSetNumCurr = DataSetNumCurr + 1;    

# Driver information
DataDriver= {'DriverList':['Kyunghan','Kyuhwan','Gyubin']};
DataDriver.update({'DriverIndex':np.concatenate((1*np.ones(103),2*np.ones(100),3*np.ones(94)))})
       

#%% Design model
# Model structure - Reccurent neural network +  MLP
# Input layer - LSTM (Output size = 100, Input size = (10,4))
#             - 1 second sequential data
#             - Inputs: Velocity, Distance, Reference acceleration, Time
# Hidden layer - MLP - dense (Output size = 10,1)
# Output layer - Softmax activation (Output size = 1)
        
K.clear_session()
model = Sequential()
model.add(SimpleRNN(ModelConfig_NumLstmUnit, return_sequences=True,input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumFeature)))
ModDens1 = Dense(1, activation='relu',input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumLstmUnit))
model.add(ModDens1)
ModReshape1 = Reshape((ModelConfig_NumInputSequence,))
model.add(ModReshape1)
ModDens2 = Dense(1, activation='relu',input_shape=(1,ModelConfig_NumInputSequence))
model.add(ModDens2)
model.compile(loss='mse', optimizer='adam')

print(model.summary())

#%% Training data set
TrainConfig_TrainRatio = 0.8;
TrainConfig_ValidRatio = 1 - TrainConfig_TrainRatio;
TrainConfig_NumTrainSet = int(DataNum*TrainConfig_TrainRatio);
TrainConfig_DataSetList = list(range(DataNum))
TrainConfig_TrainSetList = TrainConfig_DataSetList[0:TrainConfig_NumTrainSet];
TrainConfig_ValidSetList = TrainConfig_DataSetList[TrainConfig_NumTrainSet:];

[DataSet_X_Norm, DataSet_X_Max, DataSet_X_Min]= NormColArry(DataSet_X);
[DataSet_Y_Norm, DataSet_Y_Max, DataSet_Y_Min]= NormColArry(DataSet_Y);
DataNorm_Den = DataSet_X_Max-DataSet_X_Min;

x_train = DataSet_X_Norm[TrainConfig_TrainSetList,:,:];
y_train = DataSet_Y_Norm[TrainConfig_TrainSetList,:];

x_valid = DataSet_X_Norm[TrainConfig_ValidSetList,:,:];
y_valid = DataSet_Y_Norm[TrainConfig_ValidSetList,:];
#%% Fit network

#fit_history = model.fit(x_train, y_train,
#                        batch_size=ModelConfig_NumBatch, epochs=20, shuffle=True,
#                        validation_data=(x_valid, y_valid))    
#%% Save model
#model.save('RnnBraking_4dim_SimpleRNN_TwoMLP')
model = load_model('RnnBraking_4dim_SimpleRNN_TwoMLP')
#%% Determine intermediate model
layer_name = 'dense_1'
model_dense1 = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
print(model_dense1.summary())
#%% Determine intermediate model
layer_name_rnn = 'simple_rnn_1'
model_rnn = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name_rnn).output)
print(model_rnn.summary())

#%% Prediction
y_pre = model.predict(x_valid)

plt.figure
plt.plot(y_pre);
plt.plot(y_valid);
plt.show()
#%% Model validation for test case
#ValidKeyList = random.sample(DataLoad.keys(),50)
ValidKeyList = DataLoad.keys()
ResultList = []
for i in ValidKeyList:    
#    i = ValidKeyList[1];
    # Load validation case
    CaseIndex = int(i[8:])
    ValidDataSet = DataLoad[i][:,1:-1];
    ValidDataSet_Y = DataLoad[i][:,0];
    # Set the prediction range (time - input sequence time)
    PredictionRange = len(ValidDataSet) - ModelConfig_NumInputSequence;
    ValidDataSetCalc = np.zeros((ModelConfig_NumInputSequence,ModelConfig_NumFeature))
    # Set the first input (input_sequence, input_feature)
    ValidDataSetCalc[:,0:3] = ValidDataSet[:ModelConfig_NumInputSequence,0:3]    
        # Calculate reference acceleration
    ValidDataSetCalc[:,2] = -0.5*ValidDataSetCalc[:,0]*ValidDataSetCalc[:,0]/ValidDataSetCalc[:,1]    
    # Normalize sequence data
    ValidDataSetNorm = (ValidDataSetCalc - DataSet_X_Min)/DataNorm_Den
    ValidDataSet_X = np.array([ValidDataSetNorm])
    # Predict array for prediction data storage
    PredictArry  = np.zeros([PredictionRange,ModelConfig_NumFeature+1])   
    PredictArry_State  = np.zeros([PredictionRange,10])
    PredictArry_Sequence  = np.zeros([PredictionRange,10])
    for j in range(PredictionRange):
        # Calculate predicted acceleration
        Predict_Value = model.predict(ValidDataSet_X)
        Predict_Value_Rnn = model_rnn.predict(ValidDataSet_X)
        Predict_Value_Dense1 = model_dense1.predict(ValidDataSet_X)

        PredictArry_State[j,:] = np.reshape(Predict_Value_Rnn[0,-1,:],(1,10))
        PredictArry_Sequence[j,:] = np.reshape(Predict_Value_Dense1,(1,10))
        Predict_Acc = Predict_Value*(DataSet_Y_Max - DataSet_Y_Min) + DataSet_Y_Min
        # Calculate next input variables
            # Predicted velocity [0]
        Predict_Vel = ValidDataSet_X[0,-1,0]*DataNorm_Den[0] + DataSet_X_Min[0] + Predict_Acc*0.1;
        if Predict_Vel <= 0.1:
            Predict_Vel = 0.1        
            # Predicted distance [1]
        Predict_Dis = ValidDataSet_X[0,-1,1]*DataNorm_Den[1] + DataSet_X_Min[1] - Predict_Vel*0.1;
        if Predict_Dis <= 0.5:
            Predict_Dis = 0.5
            # Predicted reference acceleration [2]
        Predict_AccRef = -0.5*Predict_Vel*Predict_Vel/Predict_Dis;        
            # Process time [3]
        Predict_Time = ValidDataSet_X[0,-1,3]*DataNorm_Den[3] + DataSet_X_Min[3] + 0.1;
        # Storage predicted results
        PredictArry[j,:] = np.resize(np.array([Predict_Acc,Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time]),ModelConfig_NumFeature+1);
        # Arrange model input using predicted results
        tmpValidSetPred = np.array([Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time])
        tmpValidSetPredNorm = (tmpValidSetPred - DataSet_X_Min)/DataNorm_Den;
        ValidDataSet_X[0,0:-1,:] = ValidDataSet_X[0,1:ModelConfig_NumInputSequence,:];
        ValidDataSet_X[:,-1] = np.array([[tmpValidSetPredNorm]])
    # Save result
#    Driver = DataDriver['DriverList'][int(DataDriver['DriverIndex'][CaseIndex-1])-1]
    Driver = np.array([DataDriver['DriverIndex'][CaseIndex-1]])
    PredictData = [Driver,PredictArry,PredictArry_State,PredictArry_Sequence,DataLoad[i]]
    ResultList.append(PredictData)

mdic = {'Data':ResultList}
io.savemat('PredictResult',mdic)
#%% Plot validationresults
plt.close("all")
plt.figure(1)        
plt.plot(PredictArry[:,0], label = 'AccPredic')    
plt.plot(PredictArry[:,3], label = 'AccRefPredic')
plt.plot(ValidDataSet[ModelConfig_NumInputSequence:-1,2], label = 'AccRef')
plt.plot(ValidDataSet_Y[ModelConfig_NumInputSequence:-1], label = 'Acc')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(PredictArry_Sequence, label = 'State')

plt.show()

plt.figure(3)
plt.plot(PredictArry_State, label = 'State')
plt.legend()
plt.show()
