# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:27:00 2017

@author: Kyuhwan

 * Added Save function
 * Add freeze_graph.py
 
 revised for selective neural network
 
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K

#%%
K.clear_session()
#%%
def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    maxincolumns=np.max(data,0)
    maxinlastcolumns=maxincolumns[-1]
    return numerator / (denominator + 1e-7), maxinlastcolumns
#%%
def MinMax(data):
    return np.min(data, 0),np.max(data, 0)
#%%
def MinMaxScaler_MinMax(data,minarr,maxarr):
    numerator = data - minarr
    denominator = maxarr - minarr
    return numerator / (denominator + 1e-7)
#%%
def FindLastColLoc(data):
    for i in range(data.shape[0]-1,0,-1):
        if data[i,-1] != 0:
            Loc = i
            break
    return Loc
    
#%%

# train Parameters
NumInputSteps = 40  #NumInputSteps = NumInputSteps
Range_Predict = 10 
step_size = 10 # meter
Epoch = 5
LSTMHidOut = 128 #Hidden layer output dimension
BatchSize = 72
# Open, High, Low, Volume, Close

Rawdata_3dim = scipy.io.loadmat('RawData_EngSpd_APS_BrkPres_Loc_RadDist_RadSpd_Vel_Sel.mat')
Data_list = Rawdata_3dim['Resized_RawData3dim']
Numdataset=Data_list.shape[0]
SizeCol = Data_list.shape[1]
NumFeature = Data_list.shape[2]
#%%

#%%
pre_Min = 0
pre_Max = 0
Minarr = np.zeros(NumFeature)
Maxarr = np.zeros(NumFeature)
#%%
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
test_size = int(Numdataset * 0.2)
train_size = int(0.8*(Numdataset - test_size))
validation_size = Numdataset-test_size-train_size

DatasetList = list(range(Numdataset))
random.shuffle(DatasetList)


TestSetList = DatasetList[0:test_size]
TrainSetList = DatasetList[test_size:test_size+train_size]
ValidationSetList = DatasetList[test_size+train_size:]


TrainsetNumPer1Epoch = len(TrainSetList)
#%%
def SetXandY(data,endPnt):
    x = data
    y = data[:, [-1]]  # Close as label
    # build a dataset
    dataX = []
    dataY = []
    xc=x[:endPnt]
    yc=y[:endPnt]
    for i in range(0, len(yc)-NumInputSteps-Range_Predict):
        _x  = xc[i:i + NumInputSteps]        
        _y = yc[i +Range_Predict: i + NumInputSteps + Range_Predict]        
        dataX.append(_x)
        dataY.append(_y)
    dataX=np.array(dataX)    
    dataY=np.array(dataY)
    dataY=dataY.reshape(dataY.shape[0],dataY.shape[1])
    return dataX, dataY
        
#%%
keyDict = {"TrainingLoss","ValidationLoss"}
history_arr = dict([(key,[]) for key in keyDict])
    
#%%
model = Sequential()
model.add(LSTM(LSTMHidOut, return_sequences=True,input_shape=(NumInputSteps,NumFeature)))
model.add(LSTM(LSTMHidOut))
model.add(Dense(NumInputSteps))
model.compile(loss='mae', optimizer='adam')


# fit network
for i in range(0, Epoch*TrainsetNumPer1Epoch):
    TrainingSetSel = random.choice(TrainSetList)
    ValidationSetSel = random.choice(ValidationSetList)
    TrainX, TrainY = SetXandY(Data_normalized[TrainingSetSel,:,:],EndPntData[TrainingSetSel])
    ValX, ValY = SetXandY(Data_normalized[ValidationSetSel,:,:],EndPntData[ValidationSetSel])
    history = model.fit(TrainX, TrainY, epochs=1, batch_size=BatchSize,validation_data=(ValX, ValY), verbose=2, shuffle=False)
    history_arr['TrainingLoss'].append(history.history['loss'])
    history_arr['ValidationLoss'].append(history.history['val_loss'])

    sys.stdout.write("\rProgress: {:2.1f}%\n".format(100 * i/(Epoch*TrainsetNumPer1Epoch)))
    sys.stdout.flush()
#%%model save
model.save('ModelArchive/RadarAttached.h5')

# plot history
#%%
    
plt.close("all")
#%%
plt.figure(1)
plt.plot(history_arr['TrainingLoss'], label='train')
plt.plot(history_arr['ValidationLoss'], label='validation')
#%%
TestSetSel = random.choice(TestSetList)

TestX, TestY = SetXandY(Data_normalized[TestSetSel,:,:],EndPntData[TestSetSel])
yhat = model.predict(TestX)

#%%
testScore = math.sqrt(mean_squared_error(TestY[:,:-1], yhat[:,:-1]))
print('Test Score: %.3f RMSE' % (testScore*maxspeed))


#%%
MeasuredVelocity = TestX[:,:,-1]
#%%
Initial_vel = MeasuredVelocity[0,:-1]
Initial_vel = np.transpose(Initial_vel)
Measured_vel = np.append(Initial_vel,MeasuredVelocity[:,-1])
PlotRange=np.array(list([range(len(Measured_vel))]))

#%%
yhat_Plot= np.zeros((yhat.shape[0],Range_Predict+1))
#%%
for i in range(0,len(yhat)):
    yhat_Plot[i,:]=np.append(TestY[i,-Range_Predict-1],yhat[i,-Range_Predict:])

#%%

plt.figure(3)

for i in range(0,len(Measured_vel)-yhat.shape[1]-Range_Predict):
    plt.plot(PlotRange[0,i+yhat.shape[1]-1:i+yhat.shape[1]+Range_Predict],yhat_Plot[i,:]*maxspeed,'--',lw=0.5)

plt.plot(PlotRange[0,:],Measured_vel*maxspeed,color="black",lw=1,label='Measured')
plt.title('Estimation of velocity based on LSTM RNN')
plt.ylabel('Velocity [km/h]')
plt.xlabel('Roadway position [m]')
plt.legend()
plt.grid()
plt.show()


#%%
import pydot_ng as pydot
pydot.find_graphviz()
plot_model(model, to_file='model.png',show_shapes=True)

#%%

for layer in model.layers:
    weights = layer.get_weights()


#%%
    

#shelving or pickling my session

SaveVariables = shelve.open('ModelArchive/Test_Variables_1027.out','c') # 'n' for new
for name in['NumInputSteps','Range_Predict','step_size','Epoch','LSTMHidOut','BatchSize','MeasuredVelocity','TestSetList','Data_normalized','maxspeed','EndPntData']:
    if not name.startswith (('__','_','In','Out','exit','quit','get_ipython')):
      try:
        SaveVariables[name] = globals()[name] # I didn't undersatnd why to use globals()
      except Exception:
        pass
        print('ERROR shelving: {0}'.format(name))

SaveVariables.close()

#%%

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
#%%
frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])

tf.train.write_graph(frozen_graph, "TfModelArchive/", "Tf_Test_1027.pb", as_text=False)