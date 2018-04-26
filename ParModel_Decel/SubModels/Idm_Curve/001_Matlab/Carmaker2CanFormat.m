clear all;

load('../004_Data/CarmakerLogging/0317/CM_Ang30.mat')
% load('../004_Data/CarmakerLogging/0317/CM_Ang60.mat')
% load('../004_Data/CarmakerLogging/0317/CM_Ang90.mat')
% load('../004_Data/CarmakerLogging/0317/CM_Ang120.mat')
% load('../004_Data/CarmakerLogging/0317/CM_Ang150.mat')




%%
CM_Data = CM_Logging_RoadAng30(10);
%%
Time = CM_Data.Time.data;
GPS_Time_POS = Time;

sig_State_Lat = 180/pi*CM_Data.Car_Road_GCS_Lat.data;
sig_State_Lon = 180/pi*CM_Data.Car_Road_GCS_Long.data;
Height = CM_Data.Car_Road_GCS_Elev.data;
POS_Type = 50*ones(1,length(sig_State_Lat));
Speed = CM_Data.Car_v.data;
Direction = CM_Data.Car_Yaw.data;
LONG_ACCEL = CM_Data.Car_ax.data;
LAT_ACCEL = CM_Data.Car_ay.data;
SAS_Angle = CM_Data.Steer_WhlAng.data;
WHL_SPD_FL = 3.6*CM_Data.Car_vFL.data;
WHL_SPD_FR = 3.6*CM_Data.Car_vFR.data;
WHL_SPD_RL = 3.6*CM_Data.Car_vRL.data;
WHL_SPD_RR = 3.6*CM_Data.Car_vRR.data;
YAW_RATE = CM_Data.Car_YawRate.data;
RoadLength = CM_Data.Car_Road_sRoad.data;

Time = Time';
GPS_Time_POS = GPS_Time_POS';
sig_State_Lat = sig_State_Lat';
sig_State_Lon =sig_State_Lon';
Height = Height';
POS_Type =POS_Type';
Speed = Speed';
Direction =Direction';
LONG_ACCEL =LONG_ACCEL';
SAS_Angle = SAS_Angle';
WHL_SPD_FL = WHL_SPD_FL';
WHL_SPD_FR = WHL_SPD_FR';
WHL_SPD_RL =WHL_SPD_RL';
WHL_SPD_RR =WHL_SPD_RR';
YAW_RATE = YAW_RATE';
RoadLength = RoadLength';
LAT_ACCEL = LAT_ACCEL';




save('../004_Data/CarmakerLogging/0317/CM_CANformat_Ang30_v120.mat','sig_State_Lat','sig_State_Lon','Height','POS_Type','Speed','Direction','LONG_ACCEL','SAS_Angle','WHL_SPD_FL','WHL_SPD_FR','WHL_SPD_RL','WHL_SPD_RR','YAW_RATE','Time','GPS_Time_POS','RoadLength','LAT_ACCEL')




