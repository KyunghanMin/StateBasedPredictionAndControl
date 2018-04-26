clear all;
close all;

%%
% Kyuhwan
load('../004_Data/Amsa_RdlMdl4_YK.mat') % RoadModel

Dataset = load('../005_Amsa/Amsa_Curvature4_YK.mat');

% load('../004_Data/Midan_RdlMdl.mat') % RoadModel
% 
% Dataset = load('../006_Midan/Logging_M01.mat');
% 
% 

Totallength = (PPS_RoadSegment.breaks(end));

%% Data Preprocessing (Slicing)
Totallength = round(Totallength/10);
Totallength = 10*Totallength;
limit = Totallength; % 5100m (without traffic right), 7300m (Whole Track)
Sampling = 10; %10m


EndRoadLength = find(Dataset.RoadLength(:,end)>limit);





LenRoad = length(Dataset.RoadLength(:,end));
Ego_Velocity = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.WHL_SPD_FL(:,end));
Ego_Accel = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.LONG_ACCEL(:,end));
Engine_Speed = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.N(:,end));
Gear_Position = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.CUR_GR(:,end));
Location = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.RoadLength(:,end));
Aps = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.PV_AV_CAN(:,end));
BrakePressure = InterPolatedData(Dataset.RoadLength(:,end),  Dataset.CYL_PRES(:,end));




%% Find XYZ coordinate

XYZ= fnval(B_SplineRoadModel,Location);
X = XYZ(1,:);
Y = XYZ(2,:);
Z = XYZ(3,:);
%% Curvature Parametric 


dB_SplineRoadModel = fnder(B_SplineRoadModel);
dVal_B_SplineRoadModel = fnval(dB_SplineRoadModel,Location);
ddB_SplineRoadModel = fnder(dB_SplineRoadModel);
ddVal_B_SplineRoadModel = fnval(ddB_SplineRoadModel,Location);
crossD_DD = cross( dVal_B_SplineRoadModel',ddVal_B_SplineRoadModel');


TwoDim_dVal_B_SplineRoadModel=dVal_B_SplineRoadModel;
TwoDim_ddVal_B_SplineRoadModel=ddVal_B_SplineRoadModel;
TwoDim_dVal_B_SplineRoadModel(3,:)=0;
TwoDim_ddVal_B_SplineRoadModel(3,:) = 0;
% CrossProductEle1=TwoDim_dVal_B_SplineRoadModel(:,1).*TwoDim_ddVal_B_SplineRoadModel(:,2);
% CrossProductEle2=TwoDim_dVal_B_SplineRoadModel(:,2).*TwoDim_ddVal_B_SplineRoadModel(:,1);

TwoDim_crossD_DD = cross(TwoDim_dVal_B_SplineRoadModel',TwoDim_ddVal_B_SplineRoadModel');
% TwoDim_crossD_DD = CrossProductEle1-CrossProductEle2;

%%


numCur = sqrt([1 1 1]*abs(crossD_DD.*crossD_DD)'); % Numerator of curvature
denCur = sqrt([1 1 1]*abs(dVal_B_SplineRoadModel.*dVal_B_SplineRoadModel.*dVal_B_SplineRoadModel)); % Denominator of curvature
CurvatureRoadModel = numCur ./ denCur;



TwoDim_numCur = sqrt([1 1 1]*abs(TwoDim_crossD_DD.*TwoDim_crossD_DD)'); % Numerator of curvature
TwoDim_denCur = sqrt([1 1 1]*abs(TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel)); % Denominator of curvature
TwoDim_CurvatureRoadModel = TwoDim_numCur ./ TwoDim_denCur;





%% Find sn coordinate
sn_MaxRoadPos =limit/Sampling; %max(sn_MaxRoadPosArr);
sn_Ego_Velocity = single(zeros(1,sn_MaxRoadPos));
sn_Ego_Accel = single(zeros(1,sn_MaxRoadPos));
sn_Engine_Speed = single(zeros(1,sn_MaxRoadPos));
sn_Gear_Position = single(zeros(1,sn_MaxRoadPos));
sn_Location = single(zeros(1,sn_MaxRoadPos));
sn_Aps = single(zeros(1,sn_MaxRoadPos));
sn_BrakePressure = single(zeros(1,sn_MaxRoadPos));
sn_Curvature= single(zeros(1,sn_MaxRoadPos));
sn_X = single(zeros(1,sn_MaxRoadPos));
sn_Y = single(zeros(1,sn_MaxRoadPos));
sn_Z = single(zeros(1,sn_MaxRoadPos));
%% Arrange by RoadPosition

PrePos = 0;
count = 0;
for j = 1 : length(Dataset.RoadLength(:,end))
    CurPos  = Dataset.RoadLength(j,end);
    if CurPos<limit
        if CurPos - PrePos >Sampling-0.0001
            count = count +1;
            PrePos = CurPos;
            sn_Ego_Velocity(count) =Ego_Velocity(j);
            sn_Ego_Accel(count) = Ego_Accel(j);
            sn_Engine_Speed(count) = Engine_Speed(j);
            sn_Gear_Position(count) =Gear_Position(j);
            sn_Location(count) = Location(j);
            sn_Aps(count) = Aps(j);
            sn_BrakePressure(count) =BrakePressure(j);
            sn_Curvature(count) = TwoDim_CurvatureRoadModel(j);
            sn_X(count) = X(j);
            sn_Y(count) = Y(j);
            sn_Z(count) = Z(j);
                       
        end
    end
    
end




%%
figure(1)
subplot(2,1,1)
plot(sn_Curvature,'b','linewidth',4);
grid on;
subplot(2,1,2)
plot(sn_Ego_Velocity,'r','linewidth',4);
grid on;
%% Plot map
figure(2)
plot(sn_X,sn_Y);

%% Find High curvature
HighCurvature=find(sn_Curvature>0.005);

%% Plot map
figure(3)
plot(sn_X,sn_Y,'b','linewidth',4);
hold on;
plot(sn_X(HighCurvature),sn_Y(HighCurvature),'go','linewidth',2);


%%
HC_Curvature=sn_Curvature;
HC_Curvature(sn_Curvature<0.005)=NaN;
HC_sn_Ego_Velocity = sn_Ego_Velocity;
HC_sn_Ego_Velocity(sn_Curvature<0.005)=NaN;
figure(4)
subplot(2,1,1)
plot(sn_Curvature,'b','linewidth',4);
hold on;
plot(HC_Curvature,'go','linewidth',2);
subplot(2,1,2)
plot(sn_Ego_Velocity,'b','linewidth',4);
hold on;
plot(HC_sn_Ego_Velocity,'go','linewidth',2);

%%
FirstCurve_Entry = 60:80;
FC_Ent_Curvature = double(sn_Curvature(FirstCurve_Entry));

FC_Ent_Speed =double(sn_Ego_Velocity(FirstCurve_Entry));
FC_V0 = FC_Ent_Speed(1);


%%
fun = @(k,FC_Curvature)FC_V0-k(1)*exp(k(2)*FC_V0*FC_Curvature);
k0 = [0,0];
k = lsqcurvefit(fun,k0,FC_Ent_Curvature,FC_Ent_Speed);

%%
figure(5)
plot(FC_Ent_Speed);
hold on;
plot(fun(k,FC_Ent_Curvature));
legend('true','Curvature Model');
