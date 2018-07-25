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
from sklearn import linear_model
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Reshape
from keras.layers import SimpleRNN, LSTM
from keras import backend as K
import Idm_Lib
import random
import Idm_PredictionModel_ModelDesign_ParamOpt as OptResult
import pickle
import lib_sbpac

Color = lib_sbpac.get_ColorSet()
Color['Driver1'] = Color['BP']
Color['Driver2'] = Color['RP']
Color['Driver3'] = Color['GP']

#%% Design model
# Model structure - Reccurent neural network +  MLP
# Input layer - LSTM (Output size = 100, Input size = (10,4))
#             - 1 second sequential data
#             - Inputs: Velocity, Distance, Reference acceleration, Time
# Hidden layer - MLP - dense (Output size = 10,1)
# Output layer - Softmax activation (Output size = 1)        
def model_det(model_name, ModelStr):
    K.clear_session()
    ModelConfig_NumInputSequence = ModelStr['OptPar_InputSeq']
#    ModelConfig_NumInputSequence = 10
    ModelConfig_NumStateNum =  ModelStr['OptPar_StateNum']
#    ModelConfig_SeqAct = ModelStr['OptPar_SeqAct']
    ModelConfig_SeqAct = 'tanh'
#    ModelConfig_RecSeqAct = ModelStr['OptPar_RecSeqAct']
    ModelConfig_RecSeqAct = 'hard_sigmoid'
    ModelConfig_DenseAct = ModelStr['OptPar_DenseAct']
    ModelConfig_Optimizer = ModelStr['OptPar_Opt']
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
    globals()[model_name] = model
    print(model.summary())
    return model_name
def Cluster_AccDiff(Cri, Val):
    LowLim = Cri[0]
    UpLim = Cri[1]
    if Val <= LowLim:
        Clust = 1
    elif Val <= UpLim:
        Clust = 2
    else:
        Clust = 3
    return Clust
#%% Model configuration
ModelStr = OptResult.ModelStr
ModelConfig_NumInputSequence = ModelStr['OptPar_InputSeq']
ModelConfig_NumInputSequence = 10
ModelConfig_NumStateNum =  ModelStr['OptPar_StateNum']
ModelConfig_SeqAct = ModelStr['OptPar_SeqAct']
ModelConfig_RecSeqAct = ModelStr['OptPar_RecSeqAct']
ModelConfig_DenseAct = ModelStr['OptPar_DenseAct']
ModelConfig_Optimizer = ModelStr['OptPar_Opt']
ModelConfig_NumFeature = 4
ModelConfig_NumOutputSize = 1
ModelConfig_NumBatch = ModelStr['OptPar_BatchSize']
#%% Import data and normalization
cdir = os.getcwd()
data_dir = os.chdir('../data')

DataLoad = scipy.io.loadmat('TestUpDataVariRan.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];

os.chdir(cdir)

#with open('PredictionResult_Clust.pickle','rb') as myloaddata:
#     PredictionResult_Clust = pickle.load(myloaddata)
#     myloaddata.close()    

# Parameter arrangement with clustering
PlotIndex = {'Driver1':np.arange(1,104)}
PlotIndex['Driver2'] = np.arange(104,205)
PlotIndex['Driver3'] = np.arange(205,298)

ClusterCri = [1.2, 2.0]

ParamResult = []
Param_Cluster = {}
for key in sorted(PlotIndex.keys()):
    DriverIndex = PlotIndex[key]                    
    Param_Driver = {'MaxTp':[],'Slope':[],'MaxAcc':[],'AccDiff':[],'DataCase':[],'AccMaxPt':[],'Cluster':[],'AccDiffMaxPt':[]}
    for i in range(len(DriverIndex)):
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        VehDataArray = DataLoad[tmpDataCase]
        ParamResult_Case = Idm_Lib.RefCalc(VehDataArray)
        ParamResult.append(ParamResult_Case)
        Param_Driver['MaxTp'].append(ParamResult_Case[0]['Par_MaxPoint']/10)
        Param_Driver['Slope'].append(ParamResult_Case[0]['Par_Slope']*10)
        Param_Driver['MaxAcc'].append(ParamResult_Case[0]['Par_MaxAcc'])
        Param_Driver['AccDiff'].append(ParamResult_Case[0]['Par_AccDiff'])
        Param_Driver['AccMaxPt'].append(ParamResult_Case[0]['Par_AccMaxPt'])
        Param_Driver['AccDiffMaxPt'].append(ParamResult_Case[0]['Par_AccMaxDiff'])
        Param_Driver['DataCase'].append(tmpDataCase)
        if ParamResult_Case[0]['Par_AccMaxDiff'] > 0:
            Param_Cluster[tmpDataCase] = 1
        else:
            Param_Cluster[tmpDataCase] = 2
    glb_driver = 'Param_%s' %key
    globals()[glb_driver] = Param_Driver

#%% Driver clustering
x_index = 'AccDiff'; x_data = []
y_index = 'AccDiffMaxPt'; y_data = []
fig = plt.figure(figsize = (5,4))
fig_name = 'fig4_AccMaxPoint.png'
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(Param_Driver1[x_index],Param_Driver1[y_index],c=Color['Driver1'][5],alpha = 0.3)
ax1.scatter(Param_Driver2[x_index],Param_Driver2[y_index],c=Color['Driver2'][8],alpha = 0.3)
ax1.scatter(Param_Driver3[x_index],Param_Driver3[y_index],c=Color['Driver3'][3],alpha = 0.3)
ax1.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel(r'Deceleration slope $\alpha$ [m/s$^3$]')
plt.savefig(fig_name, format='png', dpi=600)    
#%%

for i in range(3):
    tmpData = globals()['Param_Driver%s' % (i+1)]
    x_data = x_data + tmpData[x_index]
    y_data = y_data + tmpData[y_index]
x_data = np.reshape(np.array(x_data),[-1,1])
y_data = np.reshape(np.array(y_data),[-1,1])    
plt.figure()
plt.scatter(x_data,y_data)    

regr = linear_model.LinearRegression()
regr.fit(x_data,y_data)

y_reg = regr.predict(x_data)
#%%

PlotIndex = {'Driver1':np.arange(1,104)}
PlotIndex['Driver2'] = np.arange(104,205)
PlotIndex['Driver3'] = np.arange(205,298)


fig = plt.figure(figsize = (5,4))
fig_name = 'fig4_AccMaxPoint_Regression.png'
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(Param_Driver1[x_index],Param_Driver1[y_index],c=Color['Driver1'][5],alpha = 0.3)
ax1.scatter(Param_Driver2[x_index],Param_Driver2[y_index],c=Color['Driver2'][8],alpha = 0.3)
ax1.scatter(Param_Driver3[x_index],Param_Driver3[y_index],c=Color['Driver3'][3],alpha = 0.3)
ax1.plot(x_data,y_reg)
ax1.legend(['RegressionLine','Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel(r'Acc max point [m/s$^2$]')
plt.savefig(fig_name, format='png', dpi=600)

# Clustering driver characteristics
# 1 = over case - depenssive
# 2 = under case - aggressive
ClusterIndex = 2*np.ones(297)
ClusterIndex[np.reshape(0 < y_data, 297)] = 1

for key in sorted(PlotIndex.keys()):
    DriverIndex = PlotIndex[key]
    glb_driver = 'Param_%s' %key
    globals()[glb_driver]['Cluster'] = []
    for i in range(len(DriverIndex)):
        tmpCaseIndex = DriverIndex[i]-1        
        ClusterIndex[tmpCaseIndex]        
        globals()[glb_driver]['Cluster'].append(ClusterIndex[tmpCaseIndex])

fig = plt.figure(figsize = (5,4))
#fig_name = 'fig4_AccMaxPoint_Regression.png'
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(np.array(Param_Driver1[x_index])[np.array(Param_Driver1['Cluster']) == 1],np.array(Param_Driver1[y_index])[np.array(Param_Driver1['Cluster']) == 1],c=Color['Driver1'][5],alpha = 0.3)
ax1.scatter(np.array(Param_Driver1[x_index])[np.array(Param_Driver1['Cluster']) == 2],np.array(Param_Driver1[y_index])[np.array(Param_Driver1['Cluster']) == 2],c=Color['Driver1'][5],alpha = 0.3,marker='s')

#ax1.scatter(Param_Driver2[x_index][ClusterIndex == 1],Param_Driver2[y_index][ClusterIndex == 1],c=Color['Driver2'][8],alpha = 0.3)
#ax1.scatter(Param_Driver3[x_index][ClusterIndex == 1],Param_Driver3[y_index][ClusterIndex == 1],c=Color['Driver3'][3],alpha = 0.3)
ax1.plot(x_data,y_reg)
ax1.legend(['RegressionLine','Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel(r'Acc max point [m/s$^2$]')
#plt.savefig(fig_name, format='png', dpi=600)

#%%
HistData = [Param_Driver1['Cluster'],Param_Driver2['Cluster'],Param_Driver3['Cluster']]


hist_data1 = np.array(Param_Driver1['Cluster'])
hist_data2 = np.array(Param_Driver2['Cluster'])
hist_data3 = np.array(Param_Driver3['Cluster'])

bins_set1 = np.array([0.7,1.3,1.7,2.3])
Colors = [Color['Driver1'][5],Color['Driver2'][8],Color['Driver3'][3]]
fig = plt.figure(figsize = (5,4))
fig_name = 'fig4_characteristics_histogram.png'
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
n, bins, patches = ax1.hist(HistData, bins = bins_set1, color = Colors, align = 'mid', width = 0.166)
ax1.set_xticks([0.8, 1, 1.5, 2, 2.2])
ax1.set_xticklabels(['','OverCase(Ref<Acc)','','UnderCase(Ref>Acc)',''])
ax1.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_ylabel('Case Number [-]')
plt.savefig(fig_name, format='png', dpi=600)

BlendedFac = {'Driver1':0,'Driver2':0,'Driver3':0}
BlendedFac['Driver1'] = n[0][0]/(n[0][0]+n[0][-1])
BlendedFac['Driver2'] = n[1][0]/(n[1][0]+n[1][-1])
BlendedFac['Driver3'] = n[2][0]/(n[2][0]+n[2][-1])
#%% Data clustering
# 1. Indexing -> case clustering, driver clustering, train-test clustering
# 2. (if Data == train_set), Data slicing ->  index, data slicing, slicing data clustering (all, cl1,2,3)
# 3. Normalization -> Normalization
if __name__ == '__main__':
    tmpDataSet_X = []
    tmpDataSet_Y = []
    tmpDataSet_XC1 = []
    tmpDataSet_XC2 = []
    tmpDataSet_YC1 = []
    tmpDataSet_YC2 = []
    
    for key in DataLoad.keys():
        DataCurrent = DataLoad[key][:,0:6]
        tmpDataIndex = int(key[8:])-1
        ClusterIndexCase = ClusterIndex[tmpDataIndex]
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
            if ClusterIndexCase == 1: # Over case
                tmpDataSet_XC1.append(tmpDataSet_X_array)
                tmpDataSet_YC1.append(tmpDataSet_Y_array)              
            else: # 2, Under case - aggressive
                tmpDataSet_XC2.append(tmpDataSet_X_array)
                tmpDataSet_YC2.append(tmpDataSet_Y_array)                
        del tmpDataCurrent
            
    DataSet_X = np.array(tmpDataSet_X); del tmpDataSet_X
    DataSet_Y = np.array(tmpDataSet_Y); del tmpDataSet_Y
    DataSet_X_C1 = np.array(tmpDataSet_XC1); del tmpDataSet_XC1
    DataSet_Y_C1 = np.array(tmpDataSet_YC1); del tmpDataSet_YC1
    DataSet_X_C2 = np.array(tmpDataSet_XC2); del tmpDataSet_XC2
    DataSet_Y_C2 = np.array(tmpDataSet_YC2); del tmpDataSet_YC2
    
    
        
    [DataSet_X_Norm, Data_X_Max, Data_X_Min] = Idm_Lib.NormColArry(DataSet_X)
    [DataSet_Y_Norm, Data_Y_Max, Data_Y_Min] = Idm_Lib.NormColArry(DataSet_Y)
    DataNorm_Den = Data_X_Max - Data_X_Min    
    
    [DataSet_X_C1, Data_X_Max, Data_X_Min] = Idm_Lib.NormColArry(DataSet_X_C1, 1, Data_X_Min, Data_X_Max)
    [DataSet_Y_C1, Data_Y_Max, Data_Y_Min] = Idm_Lib.NormColArry(DataSet_Y_C1, 1, Data_Y_Min, Data_Y_Max)
    [DataSet_X_C2, Data_X_Max, Data_X_Min] = Idm_Lib.NormColArry(DataSet_X_C2, 1, Data_X_Min, Data_X_Max)
    [DataSet_Y_C2, Data_Y_Max, Data_Y_Min] = Idm_Lib.NormColArry(DataSet_Y_C2, 1, Data_Y_Min, Data_Y_Max)
    
    TrainConfig_TrainRatio = 0.8
    tmp_DataLen = np.shape(DataSet_X_Norm)[0]
    
    for i in range(2):
        clindex = i + 1
        tmpVari = 'DataSet_X_C%d' % clindex
        tmpVari_y = 'DataSet_Y_C%d' % clindex
        tmpData = globals()[tmpVari]
        tmpData_y = globals()[tmpVari_y]
        tmp_DataLen = np.shape(tmpData)[0]
        data_list = list(range(tmp_DataLen))
        data_list_ran = random.shuffle(data_list)
        tmp_lstTrainSet = data_list[0:int(tmp_DataLen*TrainConfig_TrainRatio)]
        tmp_lstValidSet = data_list[int(tmp_DataLen*TrainConfig_TrainRatio):] 
        globals()['x_train_c%d' % clindex] = tmpData[tmp_lstTrainSet,:,:];
        globals()['y_train_c%d' % clindex] = tmpData_y[tmp_lstTrainSet];    
        globals()['x_valid_c%d' % clindex] = tmpData[tmp_lstValidSet,:,:];
        globals()['y_valid_c%d' % clindex] = tmpData_y[tmp_lstValidSet];
    #%% Train Model
    model_det('model_c1', ModelStr)    
    model_c1.fit(x_train_c1, y_train_c1,
              batch_size=ModelConfig_NumBatch, epochs = 130, shuffle=True,
              validation_data=(x_valid_c1, y_valid_c1))    
    model_c1.save('RnnBraking_cluster1')
    
    model_det('model_c2', ModelStr)
    model_c2.fit(x_train_c2, y_train_c2,
              batch_size=ModelConfig_NumBatch, epochs = 130, shuffle=True,
              validation_data=(x_valid_c2, y_valid_c2))    
    model_c2.save('RnnBraking_cluster2')
    #%% Model validation
    model_c1 = load_model('RnnBraking_cluster1')
    model_c2 = load_model('RnnBraking_cluster2')
    #%%
    ValidKeyList = DataLoad.keys()
    
    PredictionResult_Clust_c1 = {}
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
            Predict_Value = model_c1.predict(ValidDataSet_X)        
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
        PredictionResult_Clust_c1[i] = PredictArry
    
    PredictionResult_Clust_c2 = {}
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
            Predict_Value = model_c2.predict(ValidDataSet_X)        
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
        PredictionResult_Clust_c2[i] = PredictArry  
    PredictionResult_Clust = {}
    PredictionResult_Array = {'Over':[],'Under':[],'Blnd':[]}
#    ValidKeyList = ['CaseData232']
    for i in ValidKeyList:
        tmpDataIndex = int(i[8:])
        if tmpDataIndex < 104:
            BlendedFacCase = BlendedFac['Driver1']
        elif tmpDataIndex < 204:
            BlendedFacCase = BlendedFac['Driver2']
        else:
            BlendedFacCase = BlendedFac['Driver3']        
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
            Predict_Value_over = model_c1.predict(ValidDataSet_X)*(Data_Y_Max - Data_Y_Min) + Data_Y_Min 
            Predict_Value_under = model_c2.predict(ValidDataSet_X)*(Data_Y_Max - Data_Y_Min) + Data_Y_Min
            Predict_Acc = Predict_Value_over * BlendedFacCase + Predict_Value_under * (1 - BlendedFacCase) 
            PredictionResult_Array['Over'].append(Predict_Value_over)
            PredictionResult_Array['Under'].append(Predict_Value_under)
            PredictionResult_Array['Blnd'].append(Predict_Acc)
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
        PredictionResult_Clust[i] = PredictArry
        
    plt.close("all")    
    plt.figure(1)        
    plt.plot(PredictArry[:,0], label = 'AccPredic')    
    plt.plot(PredictArry[:,3], label = 'AccRefPredic')
    plt.plot(ValidDataSet[ModelConfig_NumInputSequence:-1,2], label = 'AccRef')
    plt.plot(ValidDataSet_Y[ModelConfig_NumInputSequence:-1], label = 'Acc')
    plt.legend()
    plt.show()
#%% Cluster calculation
#    PredictionResult_Clust = {}
    PredictionResult_Merge = {}
    for i in ValidKeyList:
        tmpDataIndex = int(i[8:])
        if tmpDataIndex < 104:
            BlendedFacCase = BlendedFac['Driver1']
        elif tmpDataIndex < 204:
            BlendedFacCase = BlendedFac['Driver2']
        else:
            BlendedFacCase = BlendedFac['Driver3']    
        PredictAcc_c1 = PredictionResult_Clust_c1[i][:,0]
        PredictAcc_c2 = PredictionResult_Clust_c2[i][:,0]
        PredictAcc = PredictAcc_c1 * BlendedFacCase + PredictAcc_c2 * (1-BlendedFacCase )
        PredictVel = np.zeros(len(PredictAcc))
        PredictDis = np.zeros(len(PredictAcc))
        PredictTime = PredictionResult_Clust_c1[i][:,4]
        PredictVel[0] = PredictionResult_Clust_c1[i][0,1]
        PredictDis[0] = PredictionResult_Clust_c1[i][0,2]
        for step in range(len(PredictAcc)-1):
            PredictVel[step+1] = PredictVel[step] + PredictAcc[step]*0.1
            PredictDis[step+1] = PredictDis[step] - PredictVel[step+1]*0.1
        PredictAccRef = -0.5*PredictVel*PredictVel/PredictDis
#        if Param_Cluster[i] == 1:
##            PredictionResult_Clust[i] = PredictionResult_Clust_c1[i]
#        else:
#            PredictionResult_Clust[i] = PredictionResult_Clust_c2[i]
        PredictionResult_Merge[i] = np.transpose(np.array(([PredictAcc,PredictVel,PredictDis,PredictAccRef,PredictTime])))
        
    with open('PredictionResult_Clust_clustering.pickle','wb') as mysavedata:
         pickle.dump(PredictionResult_Clust_c1,mysavedata)   
         pickle.dump(PredictionResult_Clust_c2,mysavedata)   
         pickle.dump(PredictionResult_Clust,mysavedata)  
         pickle.dump(PredictionResult_Merge,mysavedata)
         mysavedata.close()
    #%% 2. Prediction result for driver cases - clustering
    with open('PredictionResult.pickle','rb') as myloaddata:
         PredictionResult = pickle.load(myloaddata)
         myloaddata.close()
     
    PlotIndex = {'Driver1':[1,2,3]}
    PlotIndex['Driver2'] = [146,147,148]
    PlotIndex['Driver3'] = [232,233,236]
    
    Color['Driver1'] = Color['BP']
    Color['Driver2'] = Color['RP']
    Color['Driver3'] = Color['GP']
    
    plt.close('all')
    VelCoast = []
    for key in sorted(PlotIndex.keys()):
        print(key)    
        DriverIndex = PlotIndex[key]
        fig_name = 'fig8_%s.png' % key
        print(fig_name)
        fig = plt.figure(figsize = (8,8))
        ax1 = [];ax2 = [];ax3 = []
        fig_color = Color[key]
        for i in range(len(DriverIndex)):
            ax1.append(fig.add_subplot(2,3,1 + i))                
            ax2.append(fig.add_subplot(4,3,7 + i))
            ax3.append(fig.add_subplot(4,3,10 + i))        
            ax1[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
            ax2[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
            ax3[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
            tmpCaseIndex = DriverIndex[i]
            tmpDataCase = "CaseData%d" % tmpCaseIndex
            tmpCluIndex = Param_Cluster[tmpDataCase]
            PredicArray_c2 = PredictionResult_Clust_c2[tmpDataCase]        
            PredicArray_c1 = PredictionResult_Clust_c1[tmpDataCase]        
            PredicArray_Blended = PredictionResult_Clust[tmpDataCase]        
            PredicArray_Merge = PredictionResult_Merge[tmpDataCase]    
            PredicArray_Orig = PredictionResult[tmpDataCase]    
            TimePrediction = PredicArray_c2[:,-1]
            VehDataArray = DataLoad[tmpDataCase]
            TimeVehicle = VehDataArray[:,4] - 0.01
            tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
            VelCoast.append(VehDataArray[0,1])
#            ax1[i].plot(TimePrediction,PredicArray_c2[:,0], ls = '--', label = 'AccPredic_c2_under', c=fig_color[4])    
#            ax1[i].plot(TimePrediction,PredicArray_c1[:,0], ls = '-.',label = 'AccPredic_c1_over', c=fig_color[4])    
#            ax1[i].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=fig_color[4])    
#            ax1[i].plot(TimePrediction,PredicArray_Merge[:,0], label = 'AccPredic_merge', c=fig_color[4])    
            ax1[i].plot(TimePrediction,PredicArray_c2[:,0], ls = '--', label = 'AccPredic_c2_under', c=Color['RP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_c1[:,0], ls = '-.',label = 'AccPredic_c1_over', c=Color['RP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=Color['BP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_Merge[:,0], label = 'AccPredic_merge', c=Color['GP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_Orig[:,0], label = 'AccPredic_org', c=Color['PP'][4])    
            
            ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
#            ax1[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
    
            ax2[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', c = Color['WP'][3])
            ax2[i].plot(TimePrediction,PredicArray_Blended[:,3], label = 'AccRef_blnd', c=Color['BP'][4])    
#            ax2[i].plot(TimePrediction,PredicArray_Merge[:,3], label = 'AccRef_merge', c=Color['GP'][4])    
            ax2[i].plot(TimePrediction,PredicArray_Orig[:,3], label = 'AccRef_org', c=Color['PP'][4])    
            
            ax3[i].plot(TimePrediction,PredicArray_c2[:,2], label = 'DisPredic_blnd', c=fig_color[4])
            ax3[i].plot(TimeVehicle,VehDataArray[:,2], label = 'Dis',ls = '--', c = Color['WP'][5])           
        ylim_acc = min(ax1[0].get_ylim(), ax1[1].get_ylim(), ax1[2].get_ylim())
        ylim_vel = min(ax2[0].get_ylim(), ax2[1].get_ylim(), ax2[2].get_ylim())
        ylim_dis = min(ax3[0].get_ylim(), ax3[1].get_ylim(), ax3[2].get_ylim())    
        for i in range(len(DriverIndex)):
            tmpCaseIndex = DriverIndex[i]
            tmpDataCase = "CaseData%d" % tmpCaseIndex
            tmpCluIndex = Param_Cluster[tmpDataCase]
            ax1[i].set_ylim(ylim_acc);
            ax2[i].set_ylim(ylim_vel);
            ax3[i].set_ylim(ylim_dis);        
            title_str = 'Case_%d' % (i+1)
            ax1[i].set_title(title_str)
            ax3[i].set_xlabel('Time [s]')
            if i == 0:
                ax1[i].set_ylabel('Acceleration [m/s$^2$]')
                ax2[i].set_ylabel('Velocity [km/h]')
                ax3[i].set_ylabel('Distance [m]')
                ax1[i].legend();ax2[i].legend();ax3[i].legend();
        plt.show()      