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
from mpl_toolkits.mplot3d import Axes3D
#%% Braking data analyze
def RefCalc(data):
    LocVel = data[:,1]
    LocAcc = data[:,0]
    LocAccRef = data[:,3]
    Param_MaxAcc = -np.min(LocAcc)
    tmpAcc = 1 - LocAcc/Param_MaxAcc
    tmpAcc[tmpAcc<=0] = 0.00001
    tmpAccDiff = LocAcc - LocAccRef
    LocVelRef = LocVel/np.power(tmpAcc,0.25)
    LocVelDiff = LocVel - LocVelRef
    Param_MaxPoint = np.argmax(LocVelDiff)
#    Param_AdjPoint = np.min(np.where(tmpAccDiff[Param_MaxPoint:]<=0))
    Param_AdjPoint = np.min(np.where(np.abs(tmpAccDiff[Param_MaxPoint:])<=0.15))+Param_MaxPoint
    return Param_MaxAcc, Param_MaxPoint, Param_AdjPoint
#%% Calculate corr-coef (vector)
def CorCalc(vec_1,vec_2):
    tmpCorrMatrx = np.corrcoef(vec_1,vec_2)
    CorrCoef = tmpCorrMatrx[1][0]
    if math.isnan(CorrCoef):
        CorrCoef = 0
    return CorrCoef
#%% Cluster functions
def Clu1Pol(x_val,y_val,Cri):
    y_set = Cri['Y_Set']
    x_set = Cri['X_Set']
    CluVal = (y_set[1]-y_set[0])/(x_set[1]-x_set[0])*(x_val-x_set[1])+y_set[1]
    if y_val <= CluVal:
        CluIndex = 0
    else:
        CluIndex = 1
    return CluIndex

def CluSqr(x_val,y_val,Cri):
    x_cri = Cri['MaxPnt']
    y_cri = Cri['AccDiff']
    if x_val <= x_cri:
        if y_val<=y_cri:
            CluIndex = 0
        else:
            CluIndex = 1
    else:
        if y_val<=y_cri:
            CluIndex = 2
        else:
            CluIndex = 3
    return  CluIndex
#%% Load prediction result
with open('PredictResult.pickle','rb') as myloaddata:
    PredictData = pickle.load(myloaddata)

#%% Plot one case
NumSt = 14
NumSeq = 12
index_plot = random.choice(range(297))

PredictResult = PredictData['Predict_Int'][index_plot][0]
ValidationResult = PredictData['Predict_Int'][index_plot][3][NumSeq:,:]
AccEstClu = PredictData['Predict_Clu'][index_plot][0][:,0]
AccEst = PredictData['Predict_Int'][index_plot][0][:,0]
AccRefEst = PredictData['Predict_Int'][index_plot][0][:,3]
AccRef = PredictData['Predict_Int'][index_plot][3][NumSeq:,0]
AccRefMs = PredictData['Predict_Int'][index_plot][3][NumSeq:,3]
[tmpMaxAcc, tmpMaxPoint, tmpAdjPoint] = RefCalc(PredictResult)
plt.figure()
plt.plot(AccEst,label='Prediction')
plt.plot(AccEstClu,label='Prediction_ClusterModel')
plt.plot(AccRef,label='Reference')
plt.plot(AccRefMs,label='AccReference')
plt.plot(AccRefEst,label='AccReferenceEst')
plt.plot(tmpMaxPoint,AccRef[tmpMaxPoint],'o')
plt.plot(tmpAdjPoint,AccRef[tmpAdjPoint],'^')
plt.legend()

tmpAccDiff = AccRef - AccRefMs

#%%
Param_MaxAcc = []
Param_MaxPoint = []
Param_AdjPoint = []
Param_MaxPointAccDiff = []
Param_Driver = []
Param_StResult_MaxPoint = []
Param_StResult_AdjPoint = []
Param_SeqResult_MaxPoint = []
Param_SeqDiff_MaxPoint = []


# Prediction data
# [Predict_Acc,Predict_Vel,Predict_Dis,Predict_AccRef,Predict_Time]
# Parameter 
for i in range(len(PredictData['Predict_Int'])):
    tmpData = PredictData['Predict_Int'][i]
    Driver = PredictData['Driver'][i]
    PredictResult = tmpData[0]
    PredictState = tmpData[1]
    PredictSequence = tmpData[2]
    ValidDataSet = tmpData[3]
    [tmpMaxAcc, tmpMaxPoint, tmpAdjPoint] = RefCalc(ValidDataSet)
    Param_AdjPoint.append(tmpAdjPoint)
    Param_MaxAcc.append(tmpMaxAcc)
    Param_MaxPoint.append(tmpMaxPoint)
    Param_MaxPointAccDiff.append(ValidDataSet[tmpMaxPoint,0] - ValidDataSet[tmpMaxPoint,3])
    Param_Driver.append(Driver)
    Param_StResult_MaxPoint.append(PredictState[tmpMaxPoint-NumSeq,:])
    Param_StResult_AdjPoint.append(PredictState[tmpAdjPoint-NumSeq,:])
    Param_SeqResult_MaxPoint.append(PredictSequence[tmpMaxPoint-NumSeq,:])
#    tmpPredicData = PredictSequence[tmpMaxPoint,:]
#    tmpPredicDiff = np.zeros(NumSeq)
#    for j in range(NumSeq):
#        tmpPredicDiff[j] = tmpPredicData[-1] - tmpPredicData[j]
#    Param_SeqDiff_MaxPoint.append(tmpPredicDiff)
        
        
    
Param_MaxAcc = np.array(Param_MaxAcc)
Param_MaxPoint = np.array(Param_MaxPoint)
Param_AdjPoint = np.array(Param_AdjPoint)
Param_Driver = np.array(Param_Driver)
Param_StResult_MaxPoint = np.array(Param_StResult_MaxPoint)
Param_StResult_AdjPoint = np.array(Param_StResult_AdjPoint)
Param_SeqResult_MaxPoint = np.array(Param_SeqResult_MaxPoint)
Param_MaxPointAccDiff = np.array(Param_MaxPointAccDiff)
#%% Clustering depending on drivers
DriverSet = [1,2,3]
Param_DriverIndexLst = []
Param_MaxPointLst = []
Param_MaxPointAccDiffLst = []
Param_AdjPointLst = []
Param_MaxAccLst = []
Param_StResultLst_MaxPoint = []
Param_StResultLst_AdjPoint = []
Param_SeqResultLst_MaxPoint = []

for Driver in DriverSet:
    Param_DriverIndexLst.append(np.array([index for index,value in enumerate(Param_Driver) if value == DriverSet[np.int(Driver)-1]]))
    
    Param_MaxAccLst.append(Param_MaxAcc[np.array(Param_DriverIndexLst[Driver-1])])
    
    Param_MaxPointLst.append(Param_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
    
    Param_AdjPointLst.append(Param_AdjPoint[np.array(Param_DriverIndexLst[Driver-1])])
    
    Param_StResultLst_MaxPoint.append(Param_StResult_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
    
    Param_SeqResultLst_MaxPoint.append(Param_SeqResult_MaxPoint[np.array(Param_DriverIndexLst[Driver-1])])
    
    Param_StResultLst_AdjPoint.append(Param_StResult_AdjPoint[np.array(Param_DriverIndexLst[Driver-1])])
    
    Param_MaxPointAccDiffLst.append(Param_MaxPointAccDiff[np.array(Param_DriverIndexLst[Driver-1])])    
    
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
    tmpCorrMtrx = np.zeros([1,2,StNum])
    for node_index in range(StNum):
        tmpDataX1 = Param_MaxPointAccDiffLst[Driver-1]
        tmpDataX2 = Param_AdjPointLst[Driver-1]
        tmpDataY1 = Param_StResultLst_MaxPoint[Driver-1][:,node_index]    
        tmpDataY2 = Param_StResultLst_AdjPoint[Driver-1][:,node_index]    
        tmpCorrMtrx[0,0,node_index] = CorCalc(tmpDataX1,tmpDataY1)
        tmpCorrMtrx[0,1,node_index] = CorCalc(tmpDataX2,tmpDataY2)        
    CorrCoefLst.append(tmpCorrMtrx)
    
#%% Define maximum correlated at MaxPoint
MaxIndexPoint = []
for Driver in DriverSet:    
    tmpCorrMtrx = CorrCoefLst[Driver-1]
    MaxIndexPoint.append(np.argmax(np.abs(tmpCorrMtrx[0,:,:]),1))        
#%% Plot max point correlation between AccDiff and State values
CluCri = {'X_Set':[0.6,-0.4],'Y_Set':[40,0]}
CluRan = {'AccDiff':0.4,'MaxPnt':18}

fig, axes = plt.subplots(2, 2,figsize=(9,6))
#node_index = MaxIndexPoint[0][0]
node_index = 10
for Driver in DriverSet:
    CluIndex = []
    tmpDataX1 = Param_MaxAccLst[Driver-1]
    tmpDataX = Param_MaxPointLst[Driver-1]
    tmpDataY = Param_MaxPointAccDiffLst[Driver-1]
    tmpDataY2 = Param_StResultLst_MaxPoint[Driver-1][:,node_index]
    tmpDataZ = Param_AdjPointLst[Driver-1] - Param_MaxPointLst[Driver-1]
    for i in range(len(tmpDataX)):
        CluIndex.append(Clu1Pol(tmpDataY[i],tmpDataZ[i],CluCri))
    tmpIndexCl1 = np.array(CluIndex) == 1
    tmpIndexCl0 = np.array(CluIndex) == 0

 
#    axes[0,0].plot(tmpDataX,tmpDataY,'o',ms = 3)
#    axes[0,1].plot(tmpDataX,tmpDataX1,'o',ms = 3)
#    axes[1,1].plot(tmpDataY,tmpDataX1,'o',ms = 3)
#    axes[1,0].plot(tmpDataZ,tmpDataY,'o',ms = 3)
#    
    axes[0,0].plot(tmpDataX[tmpIndexCl1],tmpDataY[tmpIndexCl1],'o',ms = 3)
    axes[0,1].plot(tmpDataX[tmpIndexCl1],tmpDataX1[tmpIndexCl1],'o',ms = 3)
    axes[1,0].plot(tmpDataX[tmpIndexCl1],tmpDataZ[tmpIndexCl1],'o',ms = 3)
    axes[1,1].plot(tmpDataZ[tmpIndexCl1],tmpDataY[tmpIndexCl1],'o',ms = 3)
    
    axes[0,0].plot(tmpDataX[tmpIndexCl0],tmpDataY[tmpIndexCl0],'^',ms = 3)
    axes[0,1].plot(tmpDataX[tmpIndexCl0],tmpDataX1[tmpIndexCl0],'^',ms = 3)
    axes[1,0].plot(tmpDataX[tmpIndexCl0],tmpDataZ[tmpIndexCl0],'^',ms = 3)
    axes[1,1].plot(tmpDataZ[tmpIndexCl0],tmpDataY[tmpIndexCl0],'o',ms = 3)
#%% Plot max acc correlation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

node_index = MaxIndexPoint[0][0]
for Driver in DriverSet:
    tmpDataX = Param_MaxPointLst[Driver-1]
    tmpDataY = Param_StResultLst_MaxPoint[Driver-1][:,node_index]
    tmpDataZ = Param_AdjPointLst[Driver-1] - Param_MaxPointLst[Driver-1]
    ax.scatter(tmpDataX,tmpDataY,tmpDataZ)
#%% Clustering
node_index = 10
CluCri = {'X_Set':[0.6,-0.4],'Y_Set':[40,0]}
CluRan = {'AccDiff':0.4,'MaxPnt':20}
for Driver in DriverSet:
    tmpDataSet = {}
    CluIndex = []
    Data = []    
    tmpDataX = Param_MaxPointLst[Driver-1]
    tmpDataY = Param_MaxPointAccDiffLst[Driver-1]
#    tmpDataY = Param_StResultLst_MaxPoint[Driver-1][:,node_index]
    tmpDataZ = Param_AdjPointLst[Driver-1] - Param_MaxPointLst[Driver-1]
    tmpDataSet['Par_MaxPnt'] = tmpDataX
    tmpDataSet['Par_AccDif'] = tmpDataY
    tmpDataSet['Par_PntDif'] = tmpDataZ
    
    tmpDataIndex = np.array(Param_DriverIndexLst[Driver-1])
    
    for i in range(len(tmpDataX)):
        tmpCluIndex = Clu1Pol(tmpDataY[i],tmpDataZ[i],CluCri)
        if tmpCluIndex == 1:
            tmpCluIndex2 = CluSqr(tmpDataX[i],tmpDataY[i],CluRan)
            if ((tmpCluIndex2 == 2) or (tmpCluIndex2 == 0)):
                CluIndexVal = 3
            elif tmpCluIndex2 == 3:
                CluIndexVal = 2
            else:
                CluIndexVal = 1
        else:
            tmpCluIndex2 = CluSqr(tmpDataX[i],tmpDataY[i],CluRan)
            if tmpCluIndex2 >= 2:
                CluIndexVal = 2
            else:
                CluIndexVal = tmpCluIndex2
                
        CluIndex.append(CluIndexVal)
        
        tmpPredictData = np.append(np.zeros([NumSeq,5]), PredictData['Predict_Int'][tmpDataIndex[i]][0], axis = 0)
        tmpMeasureData = PredictData['Predict_Int'][tmpDataIndex[i]][3][:,0:-1]
        tmpData = np.append(tmpPredictData, tmpMeasureData, axis = 1)
        Data.append(tmpData)    
    tmpDataSet['CluIndex'] = np.array(CluIndex)    
    tmpDataSet['DataIndex'] = np.array(Param_DriverIndexLst[Driver-1])
    tmpDataSet['Data'] = np.array(Data)
    tmpDataName = 'Driver_%d' % Driver
    globals()[tmpDataName] = tmpDataSet    
#%% Import data and normalization
cdir = os.getcwd()
data_dir = os.chdir('../data')
DriverDic = {'Driver_1':Driver_1,'Driver_2':Driver_2,'Driver_3':Driver_3}
io.savemat('PreResult.mat',DriverDic)


    