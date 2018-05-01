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
import random
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
#%% Import data and normalization
cdir = os.getcwd()
data_dir = os.chdir('../data')

DataLoad = scipy.io.loadmat('TestUpDataVariRan.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];

os.chdir(cdir)

#%% Model configuration
ModelConfig_NumInputSequence = 10;
ModelConfig_NumFeature = 4;
ModelConfig_NumLstmUnit = 10;
ModelConfig_NumOutputSize = 1;
ModelConfig_NumEpoch = 30;
ModelConfig_NumBatch = 50;
#%% Cluster configuration
Cluster = {}
Cluster['MaxAccSet'] = [3,3.8,5]
Cluster['MaxPntSet'] = [83,100,80]

IndexKeySet = np.array(list(DataLoad.keys()))
IndexTestSet = np.array(random.sample(list(IndexKeySet),10))
IndexTrainSet = np.setdiff1d(IndexKeySet,IndexTestSet)
# Driver information
DataDriver= {'DriverList':['Kyunghan','Kyuhwan','Gyubin']};
DataDriver.update({'DriverIndex':np.concatenate((1*np.ones(103),2*np.ones(100),3*np.ones(94)))})
#%% Data clustering
# 1. Indexing -> case clustering, driver clustering, train-test clustering
# 2. (if Data == train_set), Data slicing ->  index, data slicing, slicing data clustering (all, cl1,2,3)
# 3. Normalization -> Normalization

ParSet = {'DriverIndex':[],'ClusterIndex':[],'MaxAcc':[],'MaxPnt':[],'Data':[]}
tmpDataSet_X = []
tmpDataSet_Y = []
tmpDataSet_XC1 = []
tmpDataSet_XC2 = []
tmpDataSet_XC3 = []
tmpDataSet_YC1 = []
tmpDataSet_YC2 = []
tmpDataSet_YC3 = []
for key in DataLoad.keys():
#    Driver indexing
    DataIndex = int(key[8:])
    DriverIndex = DataDriver['DriverIndex'][DataIndex-1]
#    Case indexing
    DataCurrent = DataLoad[key];
    [tmpMaxAcc, tmpAccRat, tmpMaxPoint] = Idm_Lib.RefCalc(DataCurrent)
    [ClustIndex, tmpDummy]= Idm_Lib.Clu_2ndPoly(tmpMaxPoint,tmpMaxAcc,Cluster)
    
    ParSet['DriverIndex'].append(DriverIndex) 
    ParSet['ClusterIndex'].append(ClustIndex)
    ParSet['MaxAcc'].append(tmpMaxAcc)
    ParSet['MaxPnt'].append(tmpMaxPoint)
    ParSet['Data'].append(key)
    
    if key in IndexTrainSet:
        tmpDataCurrent = DataCurrent
        tmpLen = np.shape(tmpDataCurrent)[0]
        tmpEndPoint = tmpLen + ModelConfig_NumInputSequence - tmpLen%ModelConfig_NumInputSequence
        DataCurrent = np.zeros((tmpEndPoint,6))
        DataCurrent[:tmpLen,:] = tmpDataCurrent
        DataCurrent[tmpLen:,:] = tmpDataCurrent[tmpLen-1,:]
        DataSize = DataCurrent.shape[0]
        for index in range(0,DataSize-ModelConfig_NumInputSequence):        
            tmpDataSet_X_array = DataCurrent[index:index+ModelConfig_NumInputSequence,1:-1] 
            tmpDataSet_Y_array = DataCurrent[index+ModelConfig_NumInputSequence,0]
            tmpDataSet_X.append(tmpDataSet_X_array)
            tmpDataSet_Y.append(tmpDataSet_Y_array)
            if ClustIndex == 1:
                tmpDataSet_XC1.append(tmpDataSet_X_array)
                tmpDataSet_YC1.append(tmpDataSet_Y_array)
            elif ClustIndex == 2:
                tmpDataSet_XC2.append(tmpDataSet_X_array)
                tmpDataSet_YC2.append(tmpDataSet_Y_array)                
            else:
                tmpDataSet_XC3.append(tmpDataSet_X_array)
                tmpDataSet_YC3.append(tmpDataSet_Y_array)                
        del tmpDataCurrent
        
DataSet_X_All = np.array(tmpDataSet_X); del tmpDataSet_X
DataSet_Y_All = np.array(tmpDataSet_Y); del tmpDataSet_Y
DataSet_X_C1 = np.array(tmpDataSet_XC1); del tmpDataSet_XC1
DataSet_Y_C1 = np.array(tmpDataSet_YC1); del tmpDataSet_YC1
DataSet_X_C2 = np.array(tmpDataSet_XC2); del tmpDataSet_XC2
DataSet_Y_C2 = np.array(tmpDataSet_YC2); del tmpDataSet_YC2
DataSet_X_C3 = np.array(tmpDataSet_XC3); del tmpDataSet_XC3
DataSet_Y_C3 = np.array(tmpDataSet_YC3); del tmpDataSet_YC3

DataSet = {}
for var in [var for var in dir() if var[0:8] == "DataSet_"]:
    DataSet[var] = globals()[var];del globals()[var]
#%% Plot cluster tendency
tmpIndex_Drv = {}    
tmpIndex_Cls = {}
tmpPlotDataX = np.array(range(5,80,1))
tmpPlotDataY = np.zeros([len(tmpPlotDataX),3])
for i in range(len(tmpPlotDataX)):  
    print(i)
    [tmpDummy, tmpPlotDataY[i,:]] = Idm_Lib.Clu_2ndPoly(tmpPlotDataX[i],tmpPlotDataX[i],Cluster)

for i in [1,2,3]:
    tmpIndex_Drv[i] = [index for index, value  in enumerate(ParSet['DriverIndex']) if value == i]
    tmpIndex_Cls[i] = [index for index, value  in enumerate(ParSet['ClusterIndex']) if value == i]


plt.figure()
plt.plot(ParSet['MaxPnt'],ParSet['MaxAcc'],'o',ms=3)
for i in [1,2,3]:
    tmpX = tmpPlotDataX
    tmpY = tmpPlotDataY[:,i-1]
    plt.plot(tmpX,tmpY) 
plt.xlim(10,70)       
plt.ylim(1,5.5)

    
plt.figure()
for i in [1,2,3]:
    tmpX = np.array(ParSet['MaxPnt'])[tmpIndex_Drv[i]]  
    tmpY = np.array(ParSet['MaxAcc'])[tmpIndex_Drv[i]]
    plt.plot(tmpX,tmpY,'o',ms = 3)    
    tmpX = tmpPlotDataX
    tmpY = tmpPlotDataY[:,i-1]
    plt.plot(tmpX,tmpY) 
    
plt.figure()
for i in [1,2,3]:
    tmpX = np.array(ParSet['MaxPnt'])[tmpIndex_Cls[i]]  
    tmpY = np.array(ParSet['MaxAcc'])[tmpIndex_Cls[i]]
    plt.plot(tmpX,tmpY,'o',ms = 3)        

#%% Normalize
[DataSet['DataSet_X_Norm_All'], Data_X_Max, Data_X_Min]= Idm_Lib.NormColArry(DataSet['DataSet_X_All'])
[DataSet['DataSet_Y_Norm_All'], Data_Y_Max, Data_Y_Min]= Idm_Lib.NormColArry(DataSet['DataSet_Y_All'])
DataNorm_Den = Data_X_Max-Data_X_Min;

for strClu in ['C1','C2','C3']:    
    DataSet['DataSet_X_Norm_' + strClu] = Idm_Lib.NormColArry(DataSet['DataSet_X_' + strClu],1,Data_X_Min,Data_X_Max)[0]
    DataSet['DataSet_Y_Norm_' + strClu] = Idm_Lib.NormColArry(DataSet['DataSet_Y_' + strClu],1,Data_Y_Min,Data_Y_Max)[0]

scipy.io.savemat('DataSetDic',DataSet)    

TrainConfig_TrainRatio = 0.8;
TrainConfig_ValidRatio = 1 - TrainConfig_TrainRatio;

for strClu in ['All','C1','C2','C3']:
    tmp_DataLen = len(DataSet['DataSet_X_Norm_' + strClu])    
    tmp_lstTrainSet = list(range(tmp_DataLen))[0:int(tmp_DataLen*TrainConfig_TrainRatio)]
    tmp_lstValidSet = list(range(tmp_DataLen))[int(tmp_DataLen*TrainConfig_TrainRatio):]
    strVarName = ['x_train_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_X_Norm_' + strClu][tmp_lstTrainSet,:,:];
    strVarName = ['x_valid_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_X_Norm_' + strClu][tmp_lstValidSet,:,:];
    strVarName = ['y_train_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_Y_Norm_' + strClu][tmp_lstTrainSet];
    strVarName = ['y_valid_' + strClu][0]
    globals()[strVarName] = DataSet['DataSet_Y_Norm_' + strClu][tmp_lstValidSet];
#%% Fit network
#K.clear_session()
#model_det('model_All')    
#fit_history_All = model_All.fit(x_train_All, y_train_All,
#                        batch_size=ModelConfig_NumBatch, epochs=ModelConfig_NumEpoch, shuffle=True,
#                        validation_data=(x_valid_All, y_valid_All))   
#model_All.save('RnnBraking_4dim_SimpleRNN_TwoMLP_all')
##%% Fit network_c1
#K.clear_session()
#model_det('model_c1')    
#fit_history_C1 = model_c1.fit(x_train_C1, y_train_C1,
#                        batch_size=ModelConfig_NumBatch, epochs=ModelConfig_NumEpoch, shuffle=True,
#                        validation_data=(x_valid_C1, y_valid_C1))   
#model_c1.save('RnnBraking_4dim_SimpleRNN_TwoMLP_C1')
##%% Fit network_c2
#K.clear_session()  
#model_det('model_c2')    
#fit_history_C2 = model_c2.fit(x_train_C2, y_train_C2,
#                        batch_size=ModelConfig_NumBatch, epochs=ModelConfig_NumEpoch, shuffle=True,
#                        validation_data=(x_valid_C2, y_valid_C2))   
#model_c2.save('RnnBraking_4dim_SimpleRNN_TwoMLP_C2')
##%% Fit network_c3  
#K.clear_session()
#model_det('model_c3')      
#fit_history_C3 = model_c3.fit(x_train_C3, y_train_C3,
#                        batch_size=ModelConfig_NumBatch, epochs=ModelConfig_NumEpoch, shuffle=True,
#                        validation_data=(x_valid_C3, y_valid_C3))   
#model_c3.save('RnnBraking_4dim_SimpleRNN_TwoMLP_C3') 
#%% Model validation for test case
#ValidKeyList = random.sample(DataLoad.keys(),50)
DicPreResult = {'Predict_Int':[],'Predict_Clu':[],'Driver':[],'Case':[]}

#model_All = load_model('RnnBraking_4dim_SimpleRNN_TwoMLP')
model_All = load_model('RnnBraking_4dim_SimpleRNN_TwoMLP_all')
model_C1 = load_model('RnnBraking_4dim_SimpleRNN_TwoMLP_C1')
model_C2 = load_model('RnnBraking_4dim_SimpleRNN_TwoMLP_C2')
model_C3 = load_model('RnnBraking_4dim_SimpleRNN_TwoMLP_C3')

for i in IndexTestSet:    
#    i = ValidKeyList[1];
    # Load validation case
    CaseIndex = int(i[8:])
    ParSetIndex = ParSet['Data'].index(i)
    ClusterIndex = ParSet['ClusterIndex'][ParSetIndex]
    DriverIndex = ParSet['DriverIndex'][ParSetIndex]
    
    DicPreResult['Driver'].append(DriverIndex)
    DicPreResult['Case'].append(ClusterIndex)
    
    ValidDataSet = DataLoad[i][:,1:-1];
    ValidDataSet_Y = DataLoad[i][:,0];
    # Set the prediction range (time - input sequence time)
    PredictionRange = len(ValidDataSet) - ModelConfig_NumInputSequence;
    ValidDataSetCalc = np.zeros((ModelConfig_NumInputSequence,ModelConfig_NumFeature))
    # Set the first input (input_sequence, input_feature)
    ValidDataSetCalc[:,0:4] = ValidDataSet[:ModelConfig_NumInputSequence,0:4]    
        # Calculate reference acceleration
    ValidDataSetCalc[:,2] = -0.5*ValidDataSetCalc[:,0]*ValidDataSetCalc[:,0]/ValidDataSetCalc[:,1]    
    # Normalize sequence data
    ValidDataSetNorm = (ValidDataSetCalc - Data_X_Min)/DataNorm_Den
    ValidDataSet_X = np.array([ValidDataSetNorm])
    # Predict array for prediction data storage
    PredictArry  = np.zeros([PredictionRange,ModelConfig_NumFeature+1])   
    PredictArry_State  = np.zeros([PredictionRange,10])
    PredictArry_Sequence  = np.zeros([PredictionRange,10])
    #------------------ Prediction using integrated model
    for j in range(PredictionRange):
        # Calculate predicted acceleration
        Predict_Value = model_All.predict(ValidDataSet_X)
#        Predict_Value_Rnn = model_rnn.predict(ValidDataSet_X)
#        Predict_Value_Dense1 = model_dense1.predict(ValidDataSet_X)
#        PredictArry_State[j,:] = np.reshape(Predict_Value_Rnn[0,-1,:],(1,10))
#        PredictArry_Sequence[j,:] = np.reshape(Predict_Value_Dense1,(1,10))
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
    PredictData = [PredictArry,PredictArry_State,PredictArry_Sequence,DataLoad[i]]
    DicPreResult['Predict_Int'].append(PredictData)
    
    #------------------ Prediction using clustering model
    ValidDataSet_X = np.array([ValidDataSetNorm])
    # Predict array for prediction data storage
    PredictArry  = np.zeros([PredictionRange,ModelConfig_NumFeature+1])   
    if ClusterIndex == 1:
        model = model_C1
    elif ClusterIndex == 2:
        model = model_C2
    else:
        model = model_C3       
        
    for j in range(PredictionRange):
        # Calculate predicted acceleration
        Predict_Value = model.predict(ValidDataSet_X)
#        Predict_Value_Rnn = model_rnn.predict(ValidDataSet_X)
#        Predict_Value_Dense1 = model_dense1.predict(ValidDataSet_X)
#        PredictArry_State[j,:] = np.reshape(Predict_Value_Rnn[0,-1,:],(1,10))
#        PredictArry_Sequence[j,:] = np.reshape(Predict_Value_Dense1,(1,10))
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
    PredictData = [PredictArry,PredictArry_State,PredictArry_Sequence,DataLoad[i]]
    DicPreResult['Predict_Clu'].append(PredictData)
