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
import Idm_Lib
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
DataSet = {}
DataNum = 0;
DataSetNum = 1;
# Data set arrange
DataSet_X_all = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumInputSequence,ModelConfig_NumFeature));
DataSet_Y_all = np.zeros((TrainConfig_NumDataSet,ModelConfig_NumOutputSize));

# Slicing and cooking
for key in DataLoad:
    DataCurrent = DataLoad[key];
    DataSize = DataCurrent.shape[0];
    DataSetNum = DataSetNum+1;
    DataSetNumCurr = 1;
    for index in range(0,DataSize-10):        
        DataSet_X_all[DataNum,:,:] = DataCurrent[index:index+ModelConfig_NumInputSequence,1:-1]
        DataSet_Y_all[DataNum,:] = DataCurrent[index+ModelConfig_NumInputSequence,0];
        DataNum = DataNum+1;
        DataSetNumCurr = DataSetNumCurr + 1;    

# Driver information
DataDriver= {'DriverList':['Kyunghan','Kyuhwan','Gyubin']};
DataDriver.update({'DriverIndex':np.concatenate((1*np.ones(103),2*np.ones(100),3*np.ones(94)))})
DataSet['DataSet_X_all'] = DataSet_X_all
DataSet['DataSet_Y_all'] = DataSet_Y_all
del DataSet_X_all
del DataSet_Y_all
#%% Data clustering
Cluster = {}
Cluster['MaxAccSet'] = [3,4,5]
Cluster['MaxPntSet'] = [60,70,80]
DataSet_X_C1 = []
DataSet_X_C2 = []
DataSet_X_C3 = []
DataSet_Y_C1 = []
DataSet_Y_C2 = []
DataSet_Y_C3 = []

for key in DataLoad:
    DataCurrent = DataLoad[key];
    DataSize = DataCurrent.shape[0];    
    # Slicing current data
    tmpDataBuff_X = np.zeros((DataSize,ModelConfig_NumInputSequence,ModelConfig_NumFeature))
    tmpDataBuff_Y = np.zeros((DataSize,ModelConfig_NumOutputSize))
    DataNum = 0
    for index in range(0,DataSize-10):        
        tmpDataBuff_X[DataNum,:,:] = DataCurrent[index:index+ModelConfig_NumInputSequence,1:-1]
        tmpDataBuff_Y[DataNum,:] = DataCurrent[index+ModelConfig_NumInputSequence,0];
        DataNum = DataNum+1;
    # Clustering
    [tmpMaxAcc, tmpAccRat, tmpMaxPoint] = Idm_Lib.RefCalc(DataCurrent)
    ClustIndex = Idm_Lib.Clu_2ndPoly(tmpMaxPoint,tmpMaxAcc,Cluster)
    if ClustIndex == 0:
        DataSet_X_C1.append(tmpDataBuff_X)
        DataSet_Y_C1.append(tmpDataBuff_Y)
    elif ClustIndex == 1:
        DataSet_X_C2.append(tmpDataBuff_X)
        DataSet_Y_C2.append(tmpDataBuff_Y)
    else:
        DataSet_X_C3.append(tmpDataBuff_X)
        DataSet_Y_C3.append(tmpDataBuff_Y)

for var in [var for var in dir() if var[0:8] == "DataSet_"]:
    DataSet[var] = np.concatenate(globals()[var]);del globals()[var]
#%% Design model
# Model structure - Reccurent neural network +  MLP
# Input layer - LSTM (Output size = 100, Input size = (10,4))
#             - 1 second sequential data
#             - Inputs: Velocity, Distance, Reference acceleration, Time
# Hidden layer - MLP - dense (Output size = 10,1)
# Output layer - Softmax activation (Output size = 1)        
def model_det(model_name):
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
    globals()[model_name] = model
    print(model.summary())
    return model_name
#%% Normalize
[DataSet['DataSet_X_Norm_all'], Data_X_Max, Data_X_Min]= Idm_Lib.NormColArry(DataSet['DataSet_X_all'])
[DataSet['DataSet_Y_Norm_all'], Data_Y_Max, Data_Y_Min]= Idm_Lib.NormColArry(DataSet['DataSet_Y_all'])
DataNorm_Den = Data_X_Max-Data_X_Min;

for strClu in ['C1','C2','C3']:    
    DataSet['DataSet_X_Norm_' + strClu] = Idm_Lib.NormColArry(DataSet['DataSet_X_' + strClu],1,Data_X_Min,Data_X_Max)[0]
    DataSet['DataSet_Y_Norm_' + strClu] = Idm_Lib.NormColArry(DataSet['DataSet_Y_' + strClu],1,Data_Y_Min,Data_Y_Max)[0]
#%%
TrainConfig_TrainRatio = 0.8;
TrainConfig_ValidRatio = 1 - TrainConfig_TrainRatio;

for strClu in ['all','C1','C2','C3']:
    tmp_DataLen = len(DataSet['DataSet_X_Norm_' + strClu])    
    tmp_lstTrainSet = list(range(tmp_DataLen))[0:int(tmp_DataLen*TrainConfig_TrainRatio)]
    tmp_lstValidSet = list(range(tmp_DataLen))[int(tmp_DataLen*TrainConfig_TrainRatio):]
    strVarName = ['x_train_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_X_Norm_' + strClu][tmp_lstTrainSet,:,:];
    strVarName = ['x_valid_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_X_Norm_' + strClu][tmp_lstValidSet,:,:];
    strVarName = ['y_train_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_Y_Norm_' + strClu][tmp_lstTrainSet,:];
    strVarName = ['y_valid_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_Y_Norm_' + strClu][tmp_lstValidSet,:];
#%% Fit network
model_det('model_all')    
fit_history_all = model_all.fit(x_train_all, y_train_all,
                        batch_size=ModelConfig_NumBatch, epochs=100, shuffle=True,
                        validation_data=(x_valid_all, y_valid_all))   
model_all.save('RnnBraking_4dim_SimpleRNN_TwoMLP_all')
#%% Fit network_c1
model_det('model_c1')    
fit_history_C1 = model_c1.fit(x_train_C1, y_train_C1,
                        batch_size=ModelConfig_NumBatch, epochs=100, shuffle=True,
                        validation_data=(x_valid_C1, y_valid_C1))   
model_c1.save('RnnBraking_4dim_SimpleRNN_TwoMLP_C1')
#%% Fit network_c2    
model_det('model_c2')    
fit_history_C2 = model_c2.fit(x_train_C2, y_train_C2,
                        batch_size=ModelConfig_NumBatch, epochs=100, shuffle=True,
                        validation_data=(x_valid_C2, y_valid_C2))   
model_c2.save('RnnBraking_4dim_SimpleRNN_TwoMLP_C2')
#%% Fit network_c3  
model_det('model_c3')      
fit_history_C3 = model_c3.fit(x_train_C3, y_train_C3,
                        batch_size=ModelConfig_NumBatch, epochs=100, shuffle=True,
                        validation_data=(x_valid_C3, y_valid_C3))   
model_c3.save('RnnBraking_4dim_SimpleRNN_TwoMLP_C3') 
#%% Save model
#model.save('RnnBraking_4dim_SimpleRNN_TwoMLP')
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
    ValidDataSetNorm = (ValidDataSetCalc - Data_X_Min)/DataNorm_Den
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
        Predict_Acc = Predict_Value*(Data_Y_Max - Data_Y_Min) + Data_Y_Min
        # Calculate next input variables
            # Predicted velocity [0]
        Predict_Vel = ValidDataSet_X[0,-1,0]*DataNorm_Den[0] + Data_X_Min[0] + Predict_Acc*0.1;
        if Predict_Vel <= 0.1:
            Predict_Vel = 0.1        
            # Predicted distance [1]
        Predict_Dis = ValidDataSet_X[0,-1,1]*DataNorm_Den[1] + Data_X_Min[1] - Predict_Vel*0.1;
        if Predict_Dis <= 0.5:
            Predict_Dis = 0.5
            # Predicted reference acceleration [2]
        Predict_AccRef = -0.5*Predict_Vel*Predict_Vel/Predict_Dis;        
            # Process time [3]
        Predict_Time = ValidDataSet_X[0,-1,3]*DataNorm_Den[3] + Data_X_Min[3] + 0.1;
        # Storage predicted results
        PredictArry[j,:] = np.resize(np.array([Predict_Acc,Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time]),ModelConfig_NumFeature+1);
        # Arrange model input using predicted results
        tmpValidSetPred = np.array([Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time])
        tmpValidSetPredNorm = (tmpValidSetPred - Data_X_Min)/DataNorm_Den;
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
