# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:30:21 2018

@author: Acelab
"""
import numpy as np
import math
#%% Declare local function
def NormColArry(data,cond=0,Min=0,Max=0):
    ''' Normalization of data array
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    '''
    if cond == False:
        MaxVal = np.max(np.max(data,0),0);
        MinVal = np.min(np.min(data,0),0);
    else:
        MaxVal = Max
        MinVal = Min
    tmpLoc_Num_NonZeroCriticValue = 1e-7;
    tmpLoc_Arry_numerator = data - MinVal;
    tmpLoc_Num_denominator = MaxVal - MinVal;    
    return tmpLoc_Arry_numerator / (tmpLoc_Num_denominator + tmpLoc_Num_NonZeroCriticValue), MaxVal, MinVal;
#%% Braking data analyze
def RefCalc(data):
    LocVel = data[:,1]
    LocAcc = data[:,0]
    LocAccRef = data[:,3];
    tmpMin = np.min(LocAcc)
    Param_MaxAcc = -tmpMin
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
#%% Cluster function
def Clu_2ndPoly(x_val,y_val,Cluster):
    y_set = np.array(Cluster['MaxAccSet'])
    x_set = np.array(Cluster['MaxPntSet'])
    tmpCluY = -y_set/x_set/x_set*x_val*x_val+y_set
    if y_val <= tmpCluY[0]:
        CluIndex = 0
    elif y_val <= tmpCluY[1]:
        CluIndex = 1
    else:
        CluIndex = 2
    return CluIndex
#%%
#cdir = os.getcwd()
#data_dir = os.chdir('../data')
#
#DataLoad = scipy.io.loadmat('TestUpDataVariRan.mat');
#del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
#TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
#del DataLoad['DataLength'];
#
#os.chdir(cdir)
#
#ModelConfig_NumInputSequence = 9
#ModelConfig_NumLstmUnit = 10
#TrainConfig_NumBatch = 50
#TrainConfig_TrainRatio = 0.8
#ModelConfig_NumFeature = 4
#ModelConfig_DataIndexNum = 6
#DataSetNum = 0
#DataNum = 0
#tmpDataSet_Y = []
#tmpDataSet_X = []
#for key in DataLoad.keys():
#    tmpDataCurrent = DataLoad[key]
#    tmpLen = np.shape(tmpDataCurrent)[0]
#    tmpEndPoint = tmpLen + ModelConfig_NumInputSequence-tmpLen%ModelConfig_NumInputSequence
#    DataCurrent = np.zeros((tmpEndPoint,ModelConfig_DataIndexNum))
#    DataCurrent[:tmpLen,:] = tmpDataCurrent
#    DataCurrent[tmpLen:,:] = tmpDataCurrent[tmpLen-1,:]
#    DataSize = DataCurrent.shape[0]
#    DataSetNum = DataSetNum+1
#    DataSetNumCurr = 1
#    for index in range(0,DataSize-10):        
#        tmpDataSet_X.append(DataCurrent[index:index+ModelConfig_NumInputSequence,1:-1])
#        tmpDataSet_Y.append(DataCurrent[index+ModelConfig_NumInputSequence,0])
#        DataNum = DataNum+1;
#        DataSetNumCurr = DataSetNumCurr + 1
#    del tmpDataCurrent    
#DataSet_X = np.array(tmpDataSet_X); del tmpDataSet_X
#DataSet_Y = np.array(tmpDataSet_Y); del tmpDataSet_Y
#    
##    
#[DataSet_X_Norm, Data_X_Max, Data_X_Min] = Idm_Lib.NormColArry(DataSet_X)
#[DataSet_Y_Norm, Data_Y_Max, Data_Y_Min] = Idm_Lib.NormColArry(DataSet_Y)
##
#tmp_DataLen = np.shape(DataSet_X_Norm)[0]
#tmp_lstTrainSet = list(range(tmp_DataLen))[0:int(tmp_DataLen*TrainConfig_TrainRatio)]
#tmp_lstValidSet = list(range(tmp_DataLen))[int(tmp_DataLen*TrainConfig_TrainRatio):] 
#
##DataNorm_Den = DataSet_X_Max-DataSet_X_Min;
#
#x_train = DataSet_X_Norm[tmp_lstTrainSet,:,:];
#y_train = DataSet_Y_Norm[tmp_lstTrainSet];
#
#x_valid = DataSet_X_Norm[tmp_lstValidSet,:,:];
#y_valid = DataSet_Y_Norm[tmp_lstValidSet];
#   
#        #//endpoint = tmpStopPoint+ModelConfig_Sequence+1-rem(tmpStopPoint,ModelConfig_Sequence);//
#K.clear_session()
#model = Sequential()
#
#ModSeq1 = SimpleRNN(ModelConfig_NumLstmUnit, return_sequences=True,input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumFeature))    
#model.add(ModSeq1)
#ModDens1 = Dense(1, activation='relu',input_shape=(ModelConfig_NumInputSequence,ModelConfig_NumLstmUnit))
#model.add(ModDens1)
#ModReshape1 = Reshape((ModelConfig_NumInputSequence,))
#model.add(ModReshape1)
#ModDens2 = Dense(1, activation='relu',input_shape=(1,ModelConfig_NumInputSequence))
#model.add(ModDens2)
#model.compile(loss='mse', optimizer='adam')
#    
#fit_history_all = model.fit(x_train, y_train,
#                        batch_size=TrainConfig_NumBatch, epochs=30, shuffle=True,
#                        validation_data=(x_valid, y_valid))    