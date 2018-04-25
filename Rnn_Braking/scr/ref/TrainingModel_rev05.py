# -*- coding: utf-8 -*-
"""
 *Revise: fixed test set
 
"""
import numpy as np
import math
import scipy.io
import random
import sys 
import matplotlib.pyplot as plt
import shelve
from keras.utils import plot_model
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, SimpleRNN
from keras import backend as K
import matplotlib.lines as mlines
from hyperopt import hp, fmin
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from itertools import combinations
from HelpFunction import MinMaxScaler_MinMax, SetXandY, FindLastColLoc, MinMax, SetXandY_Curvature, MovingAvg
import time
#%%
K.clear_session()


#%% Set hyper parameter
TPE_Iteration = 200
# train Parameters
Inputlist = [1,2,4,5,7,8,10]
#Inputlist = [10]
NumInputs = len(Inputlist)
CombiInput=combinations(Inputlist, 5)
CombiInputlist = list(CombiInput)


NumInputSteps = 20  #NumInputSteps = NumInputSteps
NumOutputSteps =15

step_size = 10 # meter
# Open, High, Low, Volume, Close


Rawdata_3dim = scipy.io.loadmat('RawData_Amsa_35.mat')
Data_list = Rawdata_3dim['RawData3dim']
Numdataset=Data_list.shape[0]
SizeCol = Data_list.shape[1]
NumFeature = Data_list.shape[2]


#%% Dataset Information
'''
0. Engine Speed
1. APS
2. BrakePressure
3. Location
4. Radar distance
5. Radar Speed
6. Curvature
7. LongAccel
8. LatAccel
9. GearPos
10. Ego Speed 
'''
#%% Radar moving average
Data_list = np.array(Data_list)
RadarDistanceIndx = 4
RadarSpdIndx = 5
for i in range(6):
    TempRadarDist=Data_list[i,:,RadarDistanceIndx]
    FilteredRadarDist=MovingAvg(TempRadarDist,3)
    Data_list[i,:,RadarDistanceIndx] = FilteredRadarDist
    
    TempRadarSpd=Data_list[i,:,RadarSpdIndx]
    FilteredRadarSpd=MovingAvg(TempRadarSpd,3)
    Data_list[i,:,RadarSpdIndx] = FilteredRadarSpd    
    
#%% Data normalization
pre_Min = 0
pre_Max = 0
Minarr = np.zeros(NumFeature)
Maxarr = np.zeros(NumFeature)

for i in range(NumFeature):
    pre_Min = 0
    pre_Max = 0
    for j  in range(Numdataset):
        Min,Max = MinMax(Data_list[j,:,i])
        if pre_Min > Min:
            pre_Min = Min
        if pre_Max < Max:
            pre_Max = Max
    Minarr[i]=(pre_Min)
    Maxarr[i]=(pre_Max)

    

Datashape = (Numdataset,SizeCol,NumFeature)
Data_normalized = np.zeros(Datashape)
for i  in range(0, Data_list.shape[0]):
    Data_normalized[i,:,:]=MinMaxScaler_MinMax(Data_list[i,:,:],Minarr,Maxarr)

#%%
maxspeed = Maxarr[-1]
#%%
EndPntData = np.zeros((Data_normalized.shape[0]))
EndPntData=EndPntData.astype(int)
for i in range(Data_normalized.shape[0]):
    EndPntData[i]=int(FindLastColLoc(Data_normalized[i,:,:]))
#%% Set Test, Train, Validation
test_size = int(Numdataset * 0.15)
train_size = int(0.84*(Numdataset - test_size))
validation_size = Numdataset-test_size-train_size

DatasetList = list(range(Numdataset))
random.shuffle(DatasetList)

TestSetList = [6,13,20,27,34]
TrainSetList = [0,1,2,3,4,7,8,9,10,11,14,15,16,17,18,21,22,23,24,25,28,29,30,31,32]
ValidationSetList = [5,12,19,26,33]

#TestSetList = DatasetList[0:test_size]
#TrainSetList = DatasetList[test_size:test_size+train_size]
#ValidationSetList = DatasetList[test_size+train_size:]
TrainsetNumPer1Epoch = len(TrainSetList)
#%% --3.1 Parameter space
space = {
#    'NumInputSteps' : hp.choice('NumInputSteps', range(2, 50, 1)),
    'inputs' : hp.choice('inputs',CombiInputlist),
    'HiddenOut1': hp.choice('HiddenOut1', range(1, 321, 4)),
    'HiddenOut2': hp.choice('HiddenOut2', range(1, 321, 4)),                
    'batch_size' : hp.choice('batch_size', range(48, 2048, 12)),
    'nb_epochs' : hp.choice('nb_epochs', range(200,1000, 10)),   
    }
#%% 
global Numiteration
Numiteration =0
Parameters=list()
MSE = list()
iterationList = list()
ChoiceInputList = list()    
keyDict = {"TrainingLoss","ValidationLoss"}
history_arr = dict([(key,[]) for key in keyDict])
def objective(params):
    start = time.time()
    global Numiteration     
    print ('\n ') 
    print ('Iteration: ', Numiteration)   
    
    model = Sequential()
#    model.add(LSTM(64, return_sequences=True,input_shape=(NumInputSteps,NumInputs)))
#    model.add(LSTM(64 ))      
    
#    model.add(LSTM(params['HiddenOut1'], return_sequences=True,input_shape=(params['NumInputSteps'],NumInputs)))    
    model.add(LSTM(params['HiddenOut1'], return_sequences=True,input_shape=(NumInputSteps,NumInputs)))
    model.add(LSTM(params['HiddenOut2'] ))          
            
    model.add(Dense(NumOutputSteps)) 
    model.compile(loss='mean_squared_error', optimizer='adam')
#    
    InputChoice = params['inputs']
#    


#    print('NumInputSteps: ', params['NumInputSteps'])
    
    #%%
    # fit network

        
    history_model = model.fit(TrainX, TrainY, epochs=1, batch_size=params['batch_size'],validation_data=(ValX, ValY), verbose=0, shuffle=False)    
 
#        
        
    #%%model save
    score=history_model.history['val_loss'][-1]
    print(' --->  MSE: ', score)
    #############################################
    Parameters.append(params)
    MSE.append(score)
    iterationList.append(Numiteration)
    ChoiceInputList.append(np.array(InputChoice))
    
    model.save('ModelArchive_Paper_VehSpd_Driver_Radar_Input200m/SpdPrd_iter(%d)_Numinput(%d).h5' %(Numiteration,NumInputs))
    Numiteration = Numiteration +1
    

    end = time.time()
    print('Elapsed time: ', end-start)
    return {'loss': score, 'status': STATUS_OK}
    


# plot history



#%%
##########################################################################################################
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=TPE_Iteration) #algo=tpe.suggest,rand.suggest
##########################################################################################################
#