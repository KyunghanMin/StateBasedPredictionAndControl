# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:43:53 2018

@author: Kyunghan
Description: Cluster algorithm development using prediction results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import math
#%% Braking data analyze
def RefCalc(data):
    LocVel = data[:,1]
    LocAcc = data[:,0]
    
    Param_MaxAcc = -np.min(LocAcc)
    tmpAcc = 1 - LocAcc/Param_MaxAcc
    tmpAcc[tmpAcc<=0] = 0.00001
   
    LocVelRef = LocVel/np.power(tmpAcc,0.25)
    LocVelDiff = LocVel - LocVelRef
    Param_MaxPoint = np.argmax(LocVelDiff)
    
    return Param_MaxAcc, Param_MaxPoint

#%% Sequence output
def SeqCor:
#%% State output
def StateCorr
   
    

#%% Load prediction result
PredictData = io.loadmat('PredictResult.mat')
Param_MaxAcc = []
Param_MaxPoint = []
Param_Driver = []
# Prediction data
# [Predict_Acc,Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time]
for i in range(len(PredictData['Data'])):
    tmpData = PredictData['Data'][i,:]
    Driver = tmpData[0]
    PredictResult = tmpData[1]
    PredictState = tmpData[2]
    PredictSequence = tmpData[3]
    [tmpMaxAcc, tmpMaxPoint] = RefCalc(PredictResult)
    Param_MaxAcc.append(tmpMaxAcc)
    Param_MaxPoint.append(tmpMaxPoint)
    Param_Driver.append(Driver)
    
    
    #%%


[x_maxA, x_maxP] = RefCalc(PredictResult);

