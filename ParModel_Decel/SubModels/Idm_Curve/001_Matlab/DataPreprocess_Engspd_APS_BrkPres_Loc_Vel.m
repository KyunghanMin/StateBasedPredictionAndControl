clear all;
close all;
%Kyuhwan
Dataset{1}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging1.mat');
Dataset{2}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging2.mat');
Dataset{3}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging3.mat');
Dataset{4}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging4.mat');
Dataset{5}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging5.mat');
Dataset{6}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging6.mat');
Dataset{7}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging7.mat');
Dataset{8}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging8.mat');
Dataset{9}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging9.mat');
Dataset{10}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Kyuhwan/Logging10.mat');
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
Dataset{11}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging1.mat');
Dataset{12}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging2.mat');
Dataset{13}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging3.mat');
Dataset{14}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging4.mat');
Dataset{15}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging5.mat');
Dataset{16}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging6.mat');
Dataset{17}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging7.mat');
Dataset{18}=load('F:\001_Project\003_Cx\999_Logging\[20171103]_[AMSA]_[MdlDb]_Mat\[20171103]\Jaewook/Logging8.mat');


%%


exTheNumofArr = 0;
LastVehiclePos = 0;
NumFeature = 5;
Numdataset = 18;



%% Data Preprocessing (Slicing)

limit = 5100; % 5100m (without traffic right), 7300m (Whole Track)
EndArr = zeros (Numdataset,1);
for i = 1 : Numdataset
    Range = find(Dataset{1, i}.RoadLength(:,1)<limit);
    EndArr(i)=Range(end);
end
% 
% (Dataset{1, i}.WHL_SPD_RR(:,2)+Dataset{1, i}.WHL_SPD_RL(:,2)+Dataset{1, i}.WHL_SPD_FR(:,2)+Dataset{1, i}.WHL_SPD_FL(:,2))/4;
%%

Ego_Velocity_Arr = single(zeros(Numdataset,8000));
Ego_Accel_Arr = single(zeros(Numdataset,8000));
Engine_Speed_Arr = single(zeros(Numdataset,8000));
Gear_Position_Arr = single(zeros(Numdataset,8000));
Location_Arr = single(zeros(Numdataset,8000));
Aps_Arr = single(zeros(Numdataset,8000));
BrakePressure_Arr = single(zeros(Numdataset,8000));

%%

Resize = 10; % [m]


AugVel = 0;
AugAcc = 0;
AugEspd = 0;
AugGearPs = 0;
AugLoc = 0;
AugAps = 0;
AugBps = 0;
for i = 1 : Numdataset
    AugVel = cat(1,AugVel,(Dataset{1, i}.WHL_SPD_RR(1:EndArr(i),1)+Dataset{1, i}.WHL_SPD_RL(1:EndArr(i),1)+Dataset{1, i}.WHL_SPD_FR(1:EndArr(i),1)+Dataset{1, i}.WHL_SPD_FL(1:EndArr(i),1))/4);
    AugAcc = cat(1,AugAcc,Dataset{1,i}.LONG_ACCEL(1:EndArr(i),1));
    AugEspd = cat(1,AugEspd,Dataset{1,i}.N(1:EndArr(i),1));
    AugGearPs = cat(1,AugGearPs,Dataset{1,i}.CUR_GR(1:EndArr(i),1));
    AugLoc = cat(1,AugLoc,Dataset{1,i}.RoadLength(1:EndArr(i),1));
    AugAps = cat(1,AugAps,Dataset{1,i}.PV_AV_CAN(1:EndArr(i),1));
    AugBps = cat(1,AugBps,Dataset{1,i}.CYL_PRES(1:EndArr(i),1));
end
Resized_AugVel=ResizeData(AugVel,Resize);
Resized_AugAcc=ResizeData(AugAcc,Resize);
Resized_AugEspd=ResizeData(AugEspd,Resize);
Resized_AugGearPs=ResizeData(AugGearPs,Resize);
Resized_AugLoc=ResizeData(AugLoc,Resize);
Resized_AugAps=ResizeData(AugAps,Resize);
Resized_AugBps=ResizeData(AugBps,Resize);


%% Arrange Ego_Velocity
for i = 1 : Numdataset
    TrackLen = fix(max(Dataset{1,i}.RoadLength(1:EndArr(i),1),[],1) - min(Dataset{1,i}.RoadLength(1:EndArr(i),1),[],1));
    LastVehiclePos_EachDataset = max(Dataset{1,i}.RoadLength(:,1),[],1);
    LastVehiclePos = LastVehiclePos+LastVehiclePos_EachDataset;
    %Velocity
    Ego_Velocity = (Dataset{1, i}.WHL_SPD_RR(1:EndArr(i),1)+Dataset{1, i}.WHL_SPD_RL(1:EndArr(i),1)+Dataset{1, i}.WHL_SPD_FR(1:EndArr(i),1)+Dataset{1, i}.WHL_SPD_FL(1:EndArr(i),1))/4;    
    Ego_Velocity_Scoord = interp1( linspace(0,1,numel(Ego_Velocity)), Ego_Velocity, linspace(0,1,TrackLen) );
    Ego_Velocity_Arr(i,1:length(Ego_Velocity_Scoord)) = Ego_Velocity_Scoord;

    %Acceleration
    Ego_Accel= Dataset{1,i}.LONG_ACCEL(1:EndArr(i),1);
    Ego_Accel_Scoord = interp1( linspace(0,1,numel(Ego_Accel)), Ego_Accel, linspace(0,1,TrackLen) );
    Ego_Accel_Arr(i,1:length(Ego_Accel_Scoord)) = Ego_Accel_Scoord;

    %Engine speed
    Engine_Speed = Dataset{1,i}.N(1:EndArr(i),1);
    Engine_Speed_Scoord = interp1( linspace(0,1,numel(Engine_Speed)), Engine_Speed, linspace(0,1,TrackLen) );
    Engine_Speed_Arr(i,1:length(Engine_Speed_Scoord)) = Engine_Speed_Scoord;

    %Gear Position
    Gear_Position = Dataset{1,i}.CUR_GR(1:EndArr(i),1);
    Gear_Position_Scoord = interp1( linspace(0,1,numel(Gear_Position)), Gear_Position, linspace(0,1,TrackLen) );
    Gear_Position_Arr(i,1:length(Gear_Position_Scoord)) = Gear_Position_Scoord;
    
    %Location Info
    Location = Dataset{1,i}.RoadLength(1:EndArr(i),1);
    Location_Scoord = interp1( linspace(0,1,numel(Location)), Location, linspace(0,1,TrackLen) );
    Location_Arr(i,1:length(Location_Scoord)) = Location_Scoord;
    %APS
    APS = Dataset{1,i}.PV_AV_CAN(1:EndArr(i),1);
    APS_Scoord = interp1( linspace(0,1,numel(APS)), APS, linspace(0,1,TrackLen) );
    Aps_Arr(i,1:length(APS_Scoord)) = APS_Scoord;    
    %BrakePressure
    BrakePressure = Dataset{1,i}.CYL_PRES(1:EndArr(i),1);
    BrakePressure_Scoord = interp1( linspace(0,1,numel(BrakePressure)), BrakePressure, linspace(0,1,TrackLen) );
    BrakePressure_Arr(i,1:length(BrakePressure_Scoord)) = BrakePressure_Scoord;       
end
% 
%%

j=1;

RawData3dim(:,:,1) = Engine_Speed_Arr;
RawData3dim(:,:,2) = Aps_Arr;
RawData3dim(:,:,3) = BrakePressure_Arr;
RawData3dim(:,:,4) = Location_Arr;
RawData3dim(:,:,5) = Ego_Velocity_Arr;

ExperimentDataset_Augmented(:,1) = Resized_AugEspd;
ExperimentDataset_Augmented(:,2) = Resized_AugAps;
ExperimentDataset_Augmented(:,3) = Resized_AugBps;
ExperimentDataset_Augmented(:,4) = Resized_AugLoc;
ExperimentDataset_Augmented(:,5) = Resized_AugVel;

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

csvwrite('F:\001_Project\003_Cx\005_SpeedPrediction\001_Keras\Experiment_Augmented_EngSpd_APS_BrkPres_Loc_Vel_Sel.csv',ExperimentDataset_Augmented);

%%

save('F:\001_Project\003_Cx\005_SpeedPrediction\001_Keras\RawData_EngSpd_APS_BrkPres_Loc_Vel_Sel.mat','Resized_RawData3dim');

