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
import pickle
import random
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
#%% Calculate corr-coef (vector)
def CorCalc(vec_1,vec_2):
    tmpCorrMatrx = np.corrcoef(vec_1,vec_2)
    CorrCoef = tmpCorrMatrx[1][0]
    if math.isnan(CorrCoef):
        CorrCoef = 0
    return CorrCoef
#%% Load prediction result
with open('PredictResult.pickle','rb') as myloaddata:
    PredictData = pickle.load(myloaddata)

NumSt = 14
NumSeq = 12
Param_MaxAcc = []
Param_MaxPoint = []
Param_Driver = []
Param_StResult_MaxPoint = []
Param_SeqResult_MaxPoint = []
Param_SeqDiff_MaxPoint = []
# Prediction data
# [Predict_Acc,Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time]
for i in range(len(PredictData['Predict_Int'])):
    tmpData = PredictData['Predict_Int'][i]
    Driver = PredictData['Driver'][i]
    PredictResult = tmpData[0]
    PredictState = tmpData[1]
    PredictSequence = tmpData[2]
    ValidDataSet = tmpData[3]
    [tmpMaxAcc, tmpMaxPoint] = RefCalc(PredictResult)
    Param_MaxAcc.append(tmpMaxAcc)
    Param_MaxPoint.append(tmpMaxPoint)
    Param_Driver.append(Driver)
    Param_StResult_MaxPoint.append(PredictState[tmpMaxPoint,:])
    Param_SeqResult_MaxPoint.append(PredictSequence[tmpMaxPoint,:])
    tmpPredicData = PredictSequence[tmpMaxPoint,:]
    tmpPredicDiff = np.zeros(NumSeq)
    for j in range(NumSeq):
        tmpPredicDiff[j] = tmpPredicData[-1] - tmpPredicData[j]
    Param_SeqDiff_MaxPoint.append(tmpPredicDiff)
        
        
    
Param_MaxAcc = np.array(Param_MaxAcc)
Param_MaxPoint = np.array(Param_MaxPoint)
Param_Driver = np.array(Param_Driver)
Param_StResult_MaxPoint = np.array(Param_StResult_MaxPoint)
Param_SeqResult_MaxPoint = np.array(Param_SeqResult_MaxPoint)
Param_SeqDiff_MaxPoint = np.array(Param_SeqDiff_MaxPoint)
#%% Plot one case
index_plot = random.choice(range(297))

PredictResult = PredictData['Predict_Int'][index_plot][0]
ValidationResult = PredictData['Predict_Int'][index_plot][3][NumSeq:,:]
AccEstClu = PredictData['Predict_Clu'][index_plot][0][:,0]
AccEst = PredictData['Predict_Int'][index_plot][0][:,0]
AccRef = PredictData['Predict_Int'][index_plot][3][NumSeq:,0]
[tmpMaxAcc, tmpMaxPoint] = RefCalc(ValidationResult)
plt.figure()
plt.plot(AccEst,label='Prediction')
plt.plot(AccEstClu,label='Prediction_ClusterModel')
plt.plot(AccRef,label='Reference')
plt.plot(tmpMaxPoint,AccRef[tmpMaxPoint],'o')
plt.legend()


#%% Clustering depending on drivers
DriverSet = [1,2,3]
Param_DriverIndexLst = []
Param_MaxPointLst = []
Param_MaxAccLst = []
Param_StResultLst_MaxPoint = []
Param_SeqResultLst_MaxPoint = []
Param_SeqDiffLst_MaxPoint = []
for Driver in DriverSet:
    Param_DriverIndexLst.append(np.array([index for index,value in enumerate(Param_Driver) if value == DriverSet[np.int(Driver)-1]]))
    Param_MaxAccLst.append(Param_MaxAcc[np.array(Param_DriverIndexLst[Driver-1])])
    Param_MaxPointLst.append(Param_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
    Param_StResultLst_MaxPoint.append(Param_StResult_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
    Param_SeqResultLst_MaxPoint.append(Param_SeqResult_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
    Param_SeqDiffLst_MaxPoint.append(Param_SeqDiff_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
#%% Calculate correleation
SeqNum = np.shape(Param_SeqResult_MaxPoint)[1];
StNum = np.shape(Param_StResult_MaxPoint)[1];
   
CorrCoefLst = []
# X indexes
#   1: Max point
#   2: Max acc
# Y indexes
#   1: State [0~9]
#   2: Seq [0~9]
#   3: Seq diff [0~9]

for Driver in DriverSet:    
    tmpCorrMtrx = np.zeros([2,3,10])
    for node_index in range(NumSeq):
        tmpDataX1 = Param_MaxPointLst[Driver-1]
        tmpDataX2 = Param_MaxAccLst[Driver-1]
        tmpDataY1 = Param_StResultLst_MaxPoint[Driver-1][:,node_index]    
        tmpDataY2 = Param_SeqResultLst_MaxPoint[Driver-1][:,node_index]    
        tmpDataY3 = Param_SeqDiffLst_MaxPoint[Driver-1][:,node_index]
        tmpCorrMtrx[0,0,node_index] = CorCalc(tmpDataX1,tmpDataY1)
        tmpCorrMtrx[0,1,node_index] = CorCalc(tmpDataX1,tmpDataY2)
        tmpCorrMtrx[0,2,node_index] = CorCalc(tmpDataX1,tmpDataY3)
        tmpCorrMtrx[1,0,node_index] = CorCalc(tmpDataX2,tmpDataY1)
        tmpCorrMtrx[1,1,node_index] = CorCalc(tmpDataX2,tmpDataY2)
        tmpCorrMtrx[1,2,node_index] = CorCalc(tmpDataX2,tmpDataY3)        
    CorrCoefLst.append(tmpCorrMtrx)
#%% Define maximum correlated at MaxPoint
MaxIndexPoint = []
for Driver in DriverSet:    
    tmpCorrMtrx = CorrCoefLst[Driver-1]
    MaxIndexPoint.append(np.argmax(np.abs(tmpCorrMtrx[0,:,:]),1))
#%% Define maximum correlated at MaxAcc
for Driver in DriverSet:    
    tmpCorrMtrx = CorrCoefLst[Driver-1]
    MaxIndexPoint.append(np.argmax(np.abs(tmpCorrMtrx[1,:,:]),1))    
#%% Plot max point correlation
index = [0,1]
fig, axes = plt.subplots(2, 3)
for i in range(2):
    node_index = MaxIndexPoint[index[i]]
    for Driver in DriverSet:
        tmpDataX1 = Param_MaxPointLst[Driver-1]
        tmpDataY = []
        tmpDataY.append(Param_StResultLst_MaxPoint[Driver-1][:,node_index[0]])    
        tmpDataY.append(Param_SeqResultLst_MaxPoint[Driver-1][:,node_index[1]])    
        tmpDataY.append(Param_SeqDiffLst_MaxPoint[Driver-1][:,node_index[2]])    
        for j in range(3):            
            axes[i,j].plot(tmpDataX1,tmpDataY[j],'o',ms = 3)
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