clear all;
close all;
%Kyuhwan
Dataset{1}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging1.mat');
Dataset{2}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging2.mat');
Dataset{3}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging3.mat');
Dataset{4}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging4.mat');
Dataset{5}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging5.mat');
Dataset{6}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging6.mat');
Dataset{7}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging7.mat');
Dataset{8}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging8.mat');
Dataset{9}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging9.mat');
Dataset{10}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Kyuhwan/Logging10.mat');
% %Kyuhwan
% Dataset{11}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging1.mat');
% Dataset{12}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging2.mat');
% Dataset{13}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging3.mat');
% Dataset{14}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging4.mat');
% Dataset{15}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging5.mat');
% Dataset{16}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging6.mat');
% Dataset{17}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging7.mat');
% Dataset{18}=load('F:\001_Project\003_Cx\999_Logging\[20171120]_[AMSA]_[MdlDb]\Matlab\Kyuhwan/Logging8.mat');
%Jaewook
Dataset{11}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging1.mat');
Dataset{12}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging2.mat');
Dataset{13}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging3.mat');
Dataset{14}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging4.mat');
Dataset{15}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging5.mat');
Dataset{16}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging6.mat');
Dataset{17}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging7.mat');
Dataset{18}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]_2\Jaewook/Logging8.mat');


%%


exTheNumofArr = 0;
LastVehiclePos = 0;

NumFeature = 7;
Numdataset = 18;



%% Data Preprocessing (Slicing)

limit = 5100; % 5100m (without traffic right), 7300m (Whole Track)
EndArr = zeros (Numdataset,1);
for i = 1 : Numdataset
    Range = find(Dataset{1, i}.RoadLength(:,end)<limit);
    EndArr(i)=Range(end);
end
%%
tEnd = zeros (Numdataset,1);
for i = 1 : Numdataset
   
    tEnd(i)=Dataset{1, i}.RoadLength( EndArr(i),1);
end

%% Radar data preprocessing 150 -> 0
for i = 1 : Numdataset
    for j = 1 : length(Dataset{1,i}.ACC_ObjDist)
        if Dataset{1,i}.ACC_ObjDist(j,end) > 146
            Dataset{1,i}.ACC_ObjDist(j,end) = 0;
        end
    end
    
end



%%

Ego_Velocity_Arr = single(zeros(Numdataset,8000));
Ego_Accel_Arr = single(zeros(Numdataset,8000));
Engine_Speed_Arr = single(zeros(Numdataset,8000));
Gear_Position_Arr = single(zeros(Numdataset,8000));
Location_Arr = single(zeros(Numdataset,8000));
Aps_Arr = single(zeros(Numdataset,8000));
BrakePressure_Arr = single(zeros(Numdataset,8000));
RadDis_Arr = single(zeros(Numdataset,8000));
RadSpd_Arr = single(zeros(Numdataset,8000));
%%

Resize = 10; % [m]



%% Arrange Ego_Velocity
for i = 1 : Numdataset
    tRoad = find(Dataset{1, i}.RoadLength(:,1)>tEnd(i));
    tRoadPnt = tRoad(1);
    TrackLen = fix(max(Dataset{1,i}.RoadLength(1:tRoadPnt,end),[],1) - min(Dataset{1,i}.RoadLength(1:tRoadPnt,end),[],1));
    LastVehiclePos_EachDataset = max(Dataset{1,i}.RoadLength(:,1),[],1);
    LastVehiclePos = LastVehiclePos+LastVehiclePos_EachDataset;

    
    %Velocity
    
    tVelocity = find(Dataset{1, i}.WHL_SPD_RR(:,1)>tEnd(i));
    tVelocityPnt = tVelocity(1);
    Ego_Velocity = (Dataset{1, i}.WHL_SPD_RR(1:tVelocity,end)+Dataset{1, i}.WHL_SPD_RL(1:tVelocity,end)+Dataset{1, i}.WHL_SPD_FR(1:tVelocity,end)+Dataset{1, i}.WHL_SPD_FL(1:tVelocity,end))/4;    
    Ego_Velocity_Scoord = interp1( linspace(0,1,numel(Ego_Velocity)), Ego_Velocity, linspace(0,1,TrackLen) );
    Ego_Velocity_Arr(i,1:length(Ego_Velocity_Scoord)) = Ego_Velocity_Scoord;

    %Acceleration
    tAccel = find(Dataset{1, i}.LONG_ACCEL(:,1)>tEnd(i));
    tAccelPnt = tAccel(1);
    Ego_Accel= Dataset{1,i}.LONG_ACCEL(1:tAccelPnt,end);
    Ego_Accel_Scoord = interp1( linspace(0,1,numel(Ego_Accel)), Ego_Accel, linspace(0,1,TrackLen) );
    Ego_Accel_Arr(i,1:length(Ego_Accel_Scoord)) = Ego_Accel_Scoord;

    %Engine speed
    tEngine_Speed = find(Dataset{1, i}.N(:,1)>tEnd(i));
    tEngine_SpeedPnt = tEngine_Speed(1);
    Engine_Speed = Dataset{1,i}.N(1:tEngine_SpeedPnt,end);
    Engine_Speed_Scoord = interp1( linspace(0,1,numel(Engine_Speed)), Engine_Speed, linspace(0,1,TrackLen) );
    Engine_Speed_Arr(i,1:length(Engine_Speed_Scoord)) = Engine_Speed_Scoord;

    %Gear Position
    tGear_Position= find(Dataset{1, i}.CUR_GR(:,1)>tEnd(i));
    tGear_PositionPnt = tGear_Position(1);
    Gear_Position = Dataset{1,i}.CUR_GR(1:tGear_PositionPnt,end);
    Gear_Position_Scoord = interp1( linspace(0,1,numel(Gear_Position)), Gear_Position, linspace(0,1,TrackLen) );
    Gear_Position_Arr(i,1:length(Gear_Position_Scoord)) = Gear_Position_Scoord;
    
    %Location Info
    tLocation= find(Dataset{1, i}.RoadLength(:,1)>tEnd(i));
    tLocationPnt = tLocation(1);
    Location = Dataset{1,i}.RoadLength(1:tLocationPnt,end);
    Location_Scoord = interp1( linspace(0,1,numel(Location)), Location, linspace(0,1,TrackLen) );
    Location_Arr(i,1:length(Location_Scoord)) = Location_Scoord;
    %APS
    tAPS= find(Dataset{1, i}.PV_AV_CAN(:,1)>tEnd(i));
    tAPSPnt = tAPS(1);
    APS = Dataset{1,i}.PV_AV_CAN(1:tAPSPnt,end);
    APS_Scoord = interp1( linspace(0,1,numel(APS)), APS, linspace(0,1,TrackLen) );
    Aps_Arr(i,1:length(APS_Scoord)) = APS_Scoord;    
    %BrakePressure
    tBrakePressure= find(Dataset{1, i}.CYL_PRES(:,1)>tEnd(i));
    tBrakePressurePnt = tBrakePressure(1);
    BrakePressure = Dataset{1,i}.CYL_PRES(1:tBrakePressurePnt,end);
    BrakePressure_Scoord = interp1( linspace(0,1,numel(BrakePressure)), BrakePressure, linspace(0,1,TrackLen) );
    BrakePressure_Arr(i,1:length(BrakePressure_Scoord)) = BrakePressure_Scoord;
    %RadarDistance
    tRadarDistance= find(Dataset{1, i}.ACC_ObjDist(:,1)>tEnd(i));
    tRadarDistancePnt = tRadarDistance(1);
    RadarDistance = Dataset{1,i}.ACC_ObjDist(1:tRadarDistancePnt,end);
    RadarDistance_Scoord = interp1( linspace(0,1,numel(RadarDistance)), RadarDistance, linspace(0,1,TrackLen) );
    RadDis_Arr(i,1:length(RadarDistance_Scoord)) = RadarDistance_Scoord;        
    %Radar Relative speed
    tRadarRelativeSpeed= find(Dataset{1, i}.ACC_ObjRelSpd(:,1)>tEnd(i));
    tRadarRelativeSpeedPnt = tRadarRelativeSpeed(1);
    RadarRelativeSpeed = Dataset{1,i}.ACC_ObjRelSpd(1:tRadarRelativeSpeedPnt,end);
    RadarRelativeSpeed_Scoord = interp1( linspace(0,1,numel(RadarRelativeSpeed)), RadarRelativeSpeed, linspace(0,1,TrackLen) );
    RadSpd_Arr(i,1:length(RadarRelativeSpeed_Scoord)) = RadarRelativeSpeed_Scoord;        
end
% 
%%

j=1;

RawData3dim(:,:,1) = Engine_Speed_Arr;
RawData3dim(:,:,2) = Aps_Arr;
RawData3dim(:,:,3) = BrakePressure_Arr;
RawData3dim(:,:,4) = Location_Arr;
RawData3dim(:,:,5) = RadDis_Arr;
RawData3dim(:,:,6) = RadSpd_Arr;
RawData3dim(:,:,7) = Ego_Velocity_Arr;

%%

Resized_RawData3dim = single(zeros(Numdataset, fix(length(Ego_Velocity_Arr)/Resize), NumFeature));
%%
for k = 1 : NumFeature
    j=1;
    for i = 1: length(Ego_Velocity_Arr)
        if mod(i,Resize)==1
            Resized_RawData3dim(:,j,k)=RawData3dim(:,i,k);
            j = j+1;
        end
    end
    
end


%%

save('F:\001_Project\003_Cx\005_SpeedPrediction\001_Keras\RawData_EngSpd_APS_BrkPres_Loc_RadDist_RadSpd_Vel_Sel.mat','Resized_RawData3dim');

