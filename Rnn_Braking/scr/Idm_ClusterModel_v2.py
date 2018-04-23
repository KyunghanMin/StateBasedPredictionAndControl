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
#%% Delete selected variables
def VarDel(string):
    VarLen = len(string)
    Var_c_list = [var for var in globals() if var[0:VarLen] == string]
    for var in range(len(Var_c_list)):
        del globals()[Var_c_list[var]]
    return Var_c_list
#%% Braking data analyze
def RefCalc(data):
    LocVel = data[:,1]
    LocAcc = data[:,0]
    LocAccRef = data[:,3]
    Param_MaxAcc = -np.min(LocAcc)
    tmpAcc = 1 - LocAcc/Param_MaxAcc
    tmpAcc[tmpAcc<=0] = 0.00001
   
    LocVelRef = LocVel/np.power(tmpAcc,0.25)
    LocVelDiff = LocVel - LocVelRef
    Param_MaxPoint = np.argmax(LocVelDiff)
    ParamAccRat_MaxPoint = LocAcc[Param_MaxPoint]/LocAccRef[Param_MaxPoint]
    return Param_MaxAcc, ParamAccRat_MaxPoint, Param_MaxPoint
#%% Calculate corr-coef (vector)
def CorCalc(vec_1,vec_2):
    tmpCorrMatrx = np.corrcoef(vec_1,vec_2)
    CorrCoef = tmpCorrMatrx[1][0]
    if math.isnan(CorrCoef):
        CorrCoef = 0
    return CorrCoef
#%% Load prediction result
PredictData = io.loadmat('PredictResult.mat')
NumSt = 10
NumSeq = 10
tmpLstMaxAcc = []
tmpLstAccRat = []
tmpLstMaxPnt = []
tmpLstVelCst = []
tmpLstAccCst = []
Driver = []
# Prediction data
# [Predict_Acc,Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time]
for i in range(len(PredictData['Data'])):
    tmpData = PredictData['Data'][i,:]
    Driver.append(np.asscalar(tmpData[0]))
    PredictResult = tmpData[1]
    PredictState = tmpData[2]
    PredictSequence = tmpData[3]
    ValidDataSet = tmpData[4]
    [tmpMaxAcc, tmpAccRat, tmpMaxPoint] = RefCalc(ValidDataSet)
    tmpLstMaxAcc.append(tmpMaxAcc)
    tmpLstAccRat.append(tmpAccRat)
    tmpLstMaxPnt.append(tmpMaxPoint)
    tmpLstVelCst.append(ValidDataSet[0,1])
    tmpLstAccCst.append(ValidDataSet[0,0] - ValidDataSet[0,3])
    
Param = {}    
Param['MaxAcc'] = np.array(tmpLstMaxAcc)
Param['AccRat'] = np.array(tmpLstAccRat)
Param['MaxPnt'] = np.array(tmpLstMaxPnt)
Param['In_VelCst'] = np.array(tmpLstVelCst)
Param['In_AccCst'] = np.array(tmpLstAccCst)
Param['Driver'] = np.array(Driver)
VarDel("tmp")             
#%% Clustering depending on drivers
DriverSet = [1,2,3]
key_list = [keys for keys in Param.keys() if not keys == 'Driver']
tmpLstDrvIndex = []
for Driver in DriverSet:
    tmpDicVal = {}
    tmpLstDrvIndex.append(np.array([index for index,value in enumerate(Param['Driver']) if value == DriverSet[np.int(Driver)-1]]))
    for key_v in key_list:
        tmpDicVal[key_v] = Param[key_v][np.array(tmpLstDrvIndex[Driver-1])]
    Param['Driver' + str(Driver)] = tmpDicVal
#VarDel("tmp")
#%% Plot clustering index
plt.close('all')
plt.figure()
for Driver in DriverSet:
    tmpDataDict = Param['Driver' + str(Driver)]
    tmpDataX = tmpDataDict['MaxPnt']
    tmpDataY = tmpDataDict['MaxAcc']
    plt.plot(tmpDataX,tmpDataY,'o',ms=3)

plt.figure()
for Driver in DriverSet:
    tmpDataDict = Param['Driver' + str(Driver)]
    tmpDataX = tmpDataDict['In_AccCst']
    tmpDataY = tmpDataDict['MaxPnt']
    plt.plot(tmpDataX,tmpDataY,'o',ms=3)

plt.figure()
for Driver in DriverSet:
    tmpDataDict = Param['Driver' + str(Driver)]
    tmpDataX = tmpDataDict['In_VelCst']
    tmpDataY = tmpDataDict['MaxAcc']
    plt.plot(tmpDataX,tmpDataY,'o',ms=3)    
#%% Plot max acc correlation
index = [4,5]
fig, axes = plt.subplots(2, 3)
for i in range(2):
    node_index = MaxIndexPoint[index[i]]
    for Driver in DriverSet:
        tmpDataX1 = Param_MaxPointLst[Driver-1]
        for j in range(3):
            tmpDataY = Param_StResultLst_MaxPoint[Driver-1][:,node_index[j]]    
            axes[i,j].plot(tmpDataX1,tmpDataY,'o',ms = 3)
#%% Plot correlation and data clustering for Maxpoint
fig, axes = plt.subplots(2, 5, figsize = (15,5))
for Driver in DriverSet:
    tmpDataX1 = Param_MaxPointLst[Driver-1]
    for node_index in range(NumSt):
        tmpDataY = Param_StResultLst_MaxPoint[Driver-1][:,node_index]
        axes[node_index//5,node_index%5-1].plot(tmpDataX1,tmpDataY,'o',ms = 3)

fig, axes = plt.subplots(2, 5, figsize = (15,5))
for Driver in DriverSet:
    tmpDataX1 = Param_MaxPointLst[Driver-1]
    for node_index in range(NumSt):
        tmpDataY = Param_SeqResultLst_MaxPoint[Driver-1][:,node_index]
        axes[node_index//5,node_index%5-1].plot(tmpDataX1,tmpDataY,'o',ms = 3)
        
fig, axes = plt.subplots(2, 5, figsize = (15,5))
for Driver in DriverSet:
    tmpDataX1 = Param_MaxPointLst[Driver-1]
    for node_index in range(NumSt):
        tmpDataY = Param_SeqDiffLst_MaxPoint[Driver-1][:,node_index]
        axes[node_index//5,node_index%5-1].plot(tmpDataX1,tmpDataY,'o',ms = 3)        