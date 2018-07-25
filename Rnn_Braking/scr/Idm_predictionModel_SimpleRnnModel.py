# -*- coding: utf-8 -*-
"""
Data: 2018.03.20
Author: Kyunghan Min (kyunghah.min@gmail.com)
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
from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape
from keras.layers import LSTM
from keras import optimizers
from keras import backend as K
import lib_sbpac
import Idm_PredictionModel_ModelDesign_ParamOpt as OptResult
import Idm_Lib
import pickle
#%% Initialization Keras session
K.clear_session()
ModelStr = OptResult.ModelStr
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

DataLoad = scipy.io.loadmat('TestUpDataVariRan.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];

os.chdir(cdir)
#%% Model configuration
ModelConfig_NumInputSequence = ModelStr['OptPar_InputSeq']
#ModelConfig_NumInputSequence = 10
ModelConfig_NumStateNum =  ModelStr['OptPar_StateNum']
#    ModelConfig_SeqAct = ModelStr['OptPar_SeqAct']
ModelConfig_SeqAct = 'tanh'
#    ModelConfig_RecSeqAct = ModelStr['OptPar_RecSeqAct']
ModelConfig_RecSeqAct = 'hard_sigmoid'
ModelConfig_DenseAct = ModelStr['OptPar_DenseAct']
ModelConfig_Optimizer = ModelStr['OptPar_Opt']
ModelConfig_NumFeature = 4
ModelConfig_NumOutputSize = 1
ModelConfig_NumBatch = ModelStr['OptPar_BatchSize']
#%% Data arrange
tmpDataSet_Y = []
tmpDataSet_X = []

for key in DataLoad.keys():
    tmpDataCurrent = DataLoad[key]
    tmpLen = np.shape(tmpDataCurrent)[0]
    tmpEndPoint = tmpLen + ModelConfig_NumInputSequence - tmpLen%ModelConfig_NumInputSequence
    DataCurrent = np.zeros((tmpEndPoint,6))
    DataCurrent[:tmpLen,:] = tmpDataCurrent
    DataCurrent[tmpLen:,:] = tmpDataCurrent[tmpLen-1,:]
    DataSize = DataCurrent.shape[0]
    for index in range(0,DataSize-ModelConfig_NumInputSequence):        
        tmpDataSet_X.append(DataCurrent[index:index+ModelConfig_NumInputSequence,1:-1])
        tmpDataSet_Y.append(DataCurrent[index+ModelConfig_NumInputSequence,0])            
    del tmpDataCurrent   
    
DataSet_X = np.array(tmpDataSet_X); del tmpDataSet_X
DataSet_Y = np.array(tmpDataSet_Y); del tmpDataSet_Y
    
    
[DataSet_X_Norm, Data_X_Max, Data_X_Min] = Idm_Lib.NormColArry(DataSet_X)
[DataSet_Y_Norm, Data_Y_Max, Data_Y_Min] = Idm_Lib.NormColArry(DataSet_Y)

TrainConfig_TrainRatio = 0.8
tmp_DataLen = np.shape(DataSet_X_Norm)[0]
data_list = list(range(tmp_DataLen))
data_list_ran = random.shuffle(data_list)
tmp_lstTrainSet = data_list[0:int(tmp_DataLen*TrainConfig_TrainRatio)]
tmp_lstValidSet = data_list[int(tmp_DataLen*TrainConfig_TrainRatio):] 

x_train = DataSet_X_Norm[tmp_lstTrainSet,:,:];
y_train = DataSet_Y_Norm[tmp_lstTrainSet];

x_valid = DataSet_X_Norm[tmp_lstValidSet,:,:];
y_valid = DataSet_Y_Norm[tmp_lstValidSet];

DataNorm_Den = Data_X_Max - Data_X_Min
#%% Model structing    
ModelConfig_NumFeature = 4
model = Sequential()
ModSeq1 = LSTM(ModelConfig_NumStateNum, activation=ModelConfig_SeqAct, recurrent_activation = ModelConfig_RecSeqAct, return_sequences=True,input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumFeature))
model.add(ModSeq1)
ModDens1 = Dense(1, activation=ModelConfig_DenseAct,input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumStateNum))
model.add(ModDens1)
ModReshape1 = Reshape((ModelConfig_NumInputSequence,))
model.add(ModReshape1)
ModDens2 = Dense(1, activation=ModelConfig_DenseAct,input_shape=(1,ModelConfig_NumInputSequence))
model.add(ModDens2)
model.compile(loss='mse', optimizer=ModelConfig_Optimizer)  
#%% Fit network
#
model.fit(x_train, y_train,
          batch_size=ModelConfig_NumBatch, epochs = 60, shuffle=True,
          validation_data=(x_valid, y_valid))    
model.save('RnnBraking_4dim')
#%% load model
#model = load_model('RnnBraking_4dim')
#%% Prediction
y_pre = model.predict(x_valid)

plt.figure()
plt.plot(y_pre);
plt.plot(y_valid);
plt.show()
#%% Model validation for test case
#ValidKeyList = random.sample(DataLoad.keys(),1)
ValidKeyList = DataLoad.keys()

PredictionResult = {}
for i in ValidKeyList:    
#    i = ValidKeyList[1];
    # Load validation case
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
    for j in range(PredictionRange):
        # Calculate predicted acceleration
        Predict_Value = model.predict(ValidDataSet_X)
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
    PredictionResult[i] = PredictArry
    
plt.close("all")
#tmpAccRef = -0.5*ValidDataSet[ModelConfig_NumInputSequence:-1,0]*ValidDataSet[ModelConfig_NumInputSequence:-1,0]/(ValidDataSet[ModelConfig_NumInputSequence-1:0,1] + 0.5);
plt.figure(1)        
plt.plot(PredictArry[:,0], label = 'AccPredic')    
plt.plot(PredictArry[:,3], label = 'AccRefPredic')
#plt.plot(tmpAccRef)
plt.plot(ValidDataSet[ModelConfig_NumInputSequence:-1,2], label = 'AccRef')
plt.plot(ValidDataSet_Y[ModelConfig_NumInputSequence:-1], label = 'Acc')
plt.legend()
plt.show()
#%%
with open('PredictionResult.pickle','wb') as mysavedata:
     pickle.dump(PredictionResult,mysavedata)   
     mysavedata.close()
#%% Plot prediction result
plt.close("all")
#tmpAccRef = -0.5*ValidDataSet[ModelConfig_NumInputSequence:-1,0]*ValidDataSet[ModelConfig_NumInputSequence:-1,0]/(ValidDataSet[ModelConfig_NumInputSequence-1:0,1] + 0.5);
plt.figure(1)        
plt.plot(PredictArry[:,0], label = 'AccPredic')    
plt.plot(PredictArry[:,3], label = 'AccRefPredic')
#plt.plot(tmpAccRef)
plt.plot(ValidDataSet[ModelConfig_NumInputSequence:-1,2], label = 'AccRef')
plt.plot(ValidDataSet_Y[ModelConfig_NumInputSequence:-1],'--', label = 'Acc')
plt.legend()
plt.show()

#%%
plt.figure(2)        
plt.plot(PredictArry[:,1])    
plt.plot(ValidDataSet[ModelConfig_NumInputSequence:-1,0])
plt.show()

plt.figure(3)        
plt.plot(PredictArry[:,2])    
plt.plot(ValidDataSet[ModelConfig_NumInputSequence:-1,1])
plt.show()

