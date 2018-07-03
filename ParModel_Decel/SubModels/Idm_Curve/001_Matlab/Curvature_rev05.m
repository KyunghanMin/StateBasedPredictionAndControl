%% File name: Curvature based Speed model
clear all;
close all;
%  Description: Design curvature based speed model
%  Authors: Kyuhwan Yeon (kyuhwanyeon@gmai.com)
%  Copyright(C) 2018 ACE Lab, All Right Reserved.
% *************************************************************************
%% Load B_spline road model & Vehicle driving data
% Kyuhwan
% load('../004_Data/Amsa_RdlMdl4_YK.mat') % RoadModel
% Dataset = load('../005_Amsa/Amsa_Curvature4_YK.mat');
% 
% load('../004_Data/Midan_RdlMdl.mat') % RoadModel
% Dataset = load('../006_Midan/Logging_M01.mat');
load('../004_Data/MidanA_RdlMdl_True.mat') % RoadModel
Dataset = load('../006_Midan/Curvature_MidanA/Logging_M03_1axis.mat'); % Vehicle driving data
%% 1. Data Preprocess
%%      [Process 1] Filtering Steering angle
Temp_StrAng = Dataset.SAS_Angle(:,end);
Length_SAS = length(Temp_StrAng);
Temp_StrAngFlt = zeros(Length_SAS,1);
for IdxData = 1: Length_SAS
    if(Temp_StrAng(IdxData) < 3276.7 && Temp_StrAng(IdxData) > 0)
        Temp_StrAngFlt(IdxData) = Temp_StrAng(IdxData);
    elseif(Temp_StrAng(IdxData) > 3276.7)
        Temp_StrAngFlt(IdxData) = (Temp_StrAng(IdxData) - 6553.6);
    end
end
for i = 1 : length(Temp_StrAngFlt)
    if (Temp_StrAngFlt(i) >200 || Temp_StrAngFlt(i) <-250)
        Temp_StrAngFlt(i) = Temp_StrAngFlt(i-1);
    end
end
%%      [Process 2] Interpolate data
Totallength = (PPS_RoadSegment.breaks(end));
Totallength = round(Totallength/10);
Totallength = 10*Totallength;
limit = Totallength; % 5100m (without traffic right), 7300m (Whole Track)
Sampling = 10; %10m

ExpandRange = 20000000;
ExpolatedRange = linspace(0,1,ExpandRange);


Roadway = single(InterPolatedData(ExpolatedRange,  (Dataset.RoadLength(:,end))));
Ego_Velocity = single(InterPolatedData(Roadway,  Dataset.WHL_SPD_FL(:,end)));
Ego_Accel = single(InterPolatedData(Roadway,  Dataset.LONG_ACCEL(:,end)));
Engine_Speed = single(InterPolatedData(Roadway,  Dataset.N(:,end)));
Gear_Position = single(InterPolatedData(Roadway,  Dataset.CUR_GR(:,end)));
Location = Roadway;
Aps = single(InterPolatedData(Roadway,  Dataset.PV_AV_CAN(:,end)));
BrakePressure = single(InterPolatedData(Roadway,  Dataset.CYL_PRES(:,end)));
SteeringAg =  single(InterPolatedData(Roadway, Temp_StrAngFlt(:,end)));

% %% Flatten coordinate (Flatten Earth)
% Lat = Dataset.sig_State_Lat(:,end);
% Long = Dataset.sig_State_Lon(:,end);
% Height = Dataset.Height(:,end);
% XYZ_flat = [];
% for i  = 1: length(Lat)
%     XYZ_flat_temp = lla2flat([Lat(i), Long(i), Height(i)], [Lat(1),Long(1)], 0,0);
%     XYZ_flat = [XYZ_flat; XYZ_flat_temp];
% end
% 
% figure(99)
% plot(XYZ_flat(:,1),XYZ_flat(:,2));
% X_flat = XYZ_flat(:,1);
% Y_flat = XYZ_flat(:,2);
% LineTemp = linspace(1,length(X_flat),length(X_flat))';
% Lines=[(1:size(X_flat,1))' (2:size(X_flat,1)+1)']; Lines(end,2)=1;
% % Lines = [LineTemp(1:end,1),LineTemp(2:end,1)+1];
% Curvature_flat = LineCurvature2D([X_flat,Y_flat],Lines);
% axis equal;
% 
% dx  = gradient(X_flat);
% ddx = gradient(dx);
% dy  = gradient(Y_flat);
% ddy = gradient(dy);
% num   = dx .* ddy - ddx .* dy;
% denom = dx .* dx + dy .* dy;
% denom = sqrt(denom);
% denom = denom .* denom .* denom;
% curvature_flat = num ./ denom;
% curvature_flat(denom < 0) = NaN;
% figure(98)
% plot(curvature_flat);
% ylim([-1,1]);
%%      [Process 3] Change to sn coordinate system
sn_MaxRoadPos =limit/Sampling; %max(sn_MaxRoadPosArr);
sn_Ego_Velocity = single(zeros(1,sn_MaxRoadPos));
sn_Ego_Accel = single(zeros(1,sn_MaxRoadPos));
sn_Engine_Speed = single(zeros(1,sn_MaxRoadPos));
sn_Gear_Position = single(zeros(1,sn_MaxRoadPos));
sn_Location = single(zeros(1,sn_MaxRoadPos));
sn_Aps = single(zeros(1,sn_MaxRoadPos));
sn_BrakePressure = single(zeros(1,sn_MaxRoadPos));
sn_SteeringAg= single(zeros(1,sn_MaxRoadPos));
% sn_X = single(zeros(1,sn_MaxRoadPos));
% sn_Y = single(zeros(1,sn_MaxRoadPos));
% sn_Z = single(zeros(1,sn_MaxRoadPos));

PrePos = 0;
count = 0;
for j = 1 : length(Roadway)
    CurPos  = Roadway(j);
    if CurPos<limit
        if CurPos - PrePos >=Sampling-0.00001
            count = count +1;
            PrePos = CurPos;
            sn_Ego_Velocity(count) =Ego_Velocity(j);
            sn_Ego_Accel(count) = Ego_Accel(j);
            sn_Engine_Speed(count) = Engine_Speed(j);
            sn_Gear_Position(count) =Gear_Position(j);
            sn_Location(count) = Location(j);
            sn_Aps(count) = Aps(j);
            sn_BrakePressure(count) =BrakePressure(j);
            sn_SteeringAg(count) =SteeringAg(j);                     
        end
    end  
end
%%      [Process 4] Find XYZ coordinate
XYZ= single(fnval(B_SplineRoadModel,sn_Location));
sn_X = XYZ(1,:);
sn_Y = XYZ(2,:);
sn_Z = XYZ(3,:);
%% 2. Calculate curvature 
%%      [Process 1] Calcuate dervative of B_Spline 
dB_SplineRoadModel = fnder(B_SplineRoadModel);
dVal_B_SplineRoadModel = single(fnval(dB_SplineRoadModel,sn_Location));
ddB_SplineRoadModel = fnder(dB_SplineRoadModel);
ddVal_B_SplineRoadModel = single(fnval(ddB_SplineRoadModel,sn_Location));
crossD_DD = cross( dVal_B_SplineRoadModel',ddVal_B_SplineRoadModel');

TwoDim_dVal_B_SplineRoadModel=dVal_B_SplineRoadModel;
TwoDim_ddVal_B_SplineRoadModel=ddVal_B_SplineRoadModel;
TwoDim_dVal_B_SplineRoadModel(3,:)=0;
TwoDim_ddVal_B_SplineRoadModel(3,:) = 0;
% CrossProductEle1=TwoDim_dVal_B_SplineRoadModel(:,1).*TwoDim_ddVal_B_SplineRoadModel(:,2);
% CrossProductEle2=TwoDim_dVal_B_SplineRoadModel(:,2).*TwoDim_ddVal_B_SplineRoadModel(:,1);
TwoDim_crossD_DD = cross(TwoDim_dVal_B_SplineRoadModel',TwoDim_ddVal_B_SplineRoadModel');
% TwoDim_crossD_DD = CrossProductEle1-CrossProductEle2;
%%      [Process 2] Calcuate Curvature (3D and 2D Curvature)
numCur = sqrt([1 1 1]*abs(crossD_DD.*crossD_DD)'); % Numerator of curvature
denCur = sqrt([1 1 1]*abs(dVal_B_SplineRoadModel.*dVal_B_SplineRoadModel.*dVal_B_SplineRoadModel)); % Denominator of curvature
denCur= denCur.^3;
ThreeDim_CurvatureRoadModel = numCur ./ denCur;

TwoDim_numCur = sqrt([1 1 1]*abs(TwoDim_crossD_DD.*TwoDim_crossD_DD)'); % Numerator of curvature
TwoDim_denCur = sqrt([1 1 1]*abs(TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel)); % Denominator of curvature
TwoDim_denCur= TwoDim_denCur.^3;
TwoDim_CurvatureRoadModel = TwoDim_numCur ./ TwoDim_denCur;


sn_Curvature = ThreeDim_CurvatureRoadModel;
flt_curvature = movmean(sn_Curvature,5);
%% 3. Plot data
%%      [Plot 1] Create Map
CurveStandard = 0.009;
HighCV_Idx=find(flt_curvature>CurveStandard);
HC_Curvature=flt_curvature;
HC_Curvature(flt_curvature<CurveStandard)=NaN;
HC_sn_Ego_Velocity = sn_Ego_Velocity;
HC_sn_Ego_Velocity(flt_curvature<CurveStandard)=NaN;
figure(3)
set(gcf,'color','w');
plot(sn_X,sn_Y,'black','linewidth',4);
hold on;
plot(sn_X(HighCV_Idx),sn_Y(HighCV_Idx),'go','linewidth',2);
hold off;
axis equal;
legend('Normal curvature', 'High curvature');
title('Map');
xlabel('[m]');
ylabel('[m]');
grid on;
%%      [Plot 2] Curvature VS Speed VS Steering angle
figure(4)
set(gcf,'color','w');
subplot(3,1,1)
plot(flt_curvature,'b','linewidth',4);
hold on;
plot(HC_Curvature,'go','linewidth',2);
hold off;
title('Curvature');
grid on;
xlabel('Roadway position [m]');

subplot(3,1,2)
plot(sn_Ego_Velocity,'b','linewidth',4);
hold on;
plot(HC_sn_Ego_Velocity,'go','linewidth',2);
hold off;
title('Speed');
grid on;
xlabel('Roadway position [m]');

subplot(3,1,3)
plot(sn_SteeringAg,'b','linewidth',4);
title('Steering Angle');
grid on;
xlabel('Roadway position [m]');
%%
CurveInit = 642/Sampling;
CurveEnd = 692/Sampling;
RangeEye2Road = 24;
Section_range = 15;
FirstCurve_Total = CurveInit:CurveEnd;
FC_Tot_Curvature = double(flt_curvature(FirstCurve_Total));
FC_Tot_Speed =double(sn_Ego_Velocity(FirstCurve_Total+RangeEye2Road));
FC_Tot_V0 = FC_Tot_Speed(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FirstCurve_Entry = CurveInit:CurveInit+Section_range;
FC_Ent_Curvature = double(flt_curvature(FirstCurve_Entry));
FC_Ent_Speed =double(sn_Ego_Velocity(FirstCurve_Entry+RangeEye2Road));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FirstCurve_middle = CurveInit+Section_range+1:CurveEnd-Section_range-1;
FC_Mid_Curvature = double(flt_curvature(FirstCurve_middle));
FC_Mid_Speed =double(sn_Ego_Velocity(FirstCurve_middle+RangeEye2Road));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FirstCurve_Exit = CurveEnd-Section_range:CurveEnd;
FC_Exit_Curvature = double(flt_curvature(FirstCurve_Exit));
FC_Exit_Speed =double(sn_Ego_Velocity(FirstCurve_Exit+RangeEye2Road));

%%
FC_Ent_V0 = FC_Ent_Speed(1);
fun1 = @(k,FC_Curvature)FC_Ent_V0-k(1)*exp(k(2)*FC_Ent_V0*FC_Curvature);
k0 = [0,0];
k_ent = lsqcurvefit(fun1,k0,FC_Ent_Curvature,FC_Ent_Speed);
CurMdlSpd_Ent = fun1(k_ent,FC_Ent_Curvature);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FC_Mid_V0 = CurMdlSpd_Ent(end);
fun2 = @(k,FC_Curvature)FC_Mid_V0-k(1)*exp(k(2)*FC_Mid_V0*FC_Curvature);
k_mid = lsqcurvefit(fun2,k0,FC_Mid_Curvature,FC_Mid_Speed);
CurMdlSpd_Mid = fun2(k_mid,FC_Mid_Curvature);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FC_Exit_V0 = CurMdlSpd_Mid(end);
fun3 = @(k,FC_Curvature)FC_Exit_V0-k(1)*exp(k(2)*FC_Exit_V0*FC_Curvature);
k_exit = lsqcurvefit(fun3,k0,FC_Exit_Curvature,FC_Exit_Speed);
CurMdlSpd_Exit = fun3(k_exit,FC_Exit_Curvature);

%%
figure(5)
subplot(3,1,1)
plot(FC_Ent_Speed);
hold on;
plot(fun1(k_ent,FC_Ent_Curvature));
legend('true','Curvature Model');
hold off;
subplot(3,1,2)
plot(FC_Mid_Speed);
hold on;
plot(fun2(k_mid,FC_Mid_Curvature));
legend('true','Curvature Model');
hold off;
subplot(3,1,3)
plot(FC_Exit_Speed);
hold on;
plot(fun3(k_exit,FC_Exit_Curvature));
legend('true','Curvature Model');
hold off;
%%
CurvMdlSpd = [CurMdlSpd_Ent CurMdlSpd_Mid CurMdlSpd_Exit];
figure(6)
plot(FC_Tot_Speed);
hold on;
plot(CurvMdlSpd);
hold off;
hold on;
hold off;



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[FC_Curve_max,FC_Curve_maxIdx] = max(FC_Tot_Curvature);
Section_range = FC_Curve_maxIdx;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FirstCurve_Entry = CurveInit:CurveInit+Section_range;
FC_Ent_Curvature = double(flt_curvature(FirstCurve_Entry));
FC_Ent_Speed =double(sn_Ego_Velocity(FirstCurve_Entry+RangeEye2Road));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FirstCurve_Exit = CurveInit+Section_range+1:CurveEnd;
FC_Exit_Curvature = double(flt_curvature(FirstCurve_Exit));
FC_Exit_Speed =double(sn_Ego_Velocity(FirstCurve_Exit+RangeEye2Road));

%%
FC_Ent_V0 = FC_Ent_Speed(1);
fun1 = @(k,FC_Curvature)FC_Ent_V0-k(1)*exp(k(2)*FC_Ent_V0*FC_Curvature);
k0 = [0,0];
k_ent = lsqcurvefit(fun1,k0,FC_Ent_Curvature,FC_Ent_Speed);
CurMdlSpd_Ent = fun1(k_ent,FC_Ent_Curvature);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FC_Exit_V0 = CurMdlSpd_Ent(end);
fun3 = @(k,FC_Curvature)FC_Exit_V0-k(1)*exp(k(2)*FC_Exit_V0*FC_Curvature);
k_exit = lsqcurvefit(fun3,k0,FC_Exit_Curvature,FC_Exit_Speed);
CurMdlSpd_Exit = fun3(k_exit,FC_Exit_Curvature);



%%
figure(7)
subplot(2,1,1)
plot(FC_Ent_Speed);
hold on;
hold off;
plot(fun1(k_ent,FC_Ent_Curvature));
legend('true','Curvature Model');
subplot(2,1,2)
plot(FC_Exit_Speed);
hold on;
plot(fun3(k_exit,FC_Exit_Curvature));
hold off;
legend('true','Curvature Model');
%%
CurvMdlSpd = [CurMdlSpd_Ent CurMdlSpd_Exit];
figure(8)
plot(FC_Tot_Speed);
hold on;
plot(CurvMdlSpd);
hold on;
hold off;


%% Calculate radius
Radius = [];
for i  = 1:length(sn_Location)-2
    x1=sn_X(i);
    y1=sn_Y(i);
    x2=sn_X(i+1);
    y2=sn_Y(i+1);
    x3=sn_X(i+2);
    y3=sn_Y(i+2);
    
    input=[x1 y1; x2 y2; x3 y3];
    [R,xyrc] = fit_circle_through_3_points(input);
    Radius=[Radius R];
end

FC_Radius = Radius(FirstCurve_Total+1);
figure(8)

subplot(2,1,1)

plot(1./Radius);
legend('1/Radius');
title('Comparison Radius and Curvature');
subplot(2,1,2)
plot(sn_Curvature);
legend('Curvature');
figure(9)
plot(FC_Radius);
ylim([0,100]);


%%
LatMax = 3.8;
Spd_Rad = sqrt(LatMax*FC_Radius)*3.6;
figure(10)
plot(Spd_Rad);
%%
Small_RadIdx = find(FC_Radius<30);
Spd_SmallRad = sqrt(LatMax*FC_Radius(Small_RadIdx))*3.6;
figure(11)
plot(Spd_SmallRad);


% %%
% figure(12)
% subplot(2,1,1)
% plot(Radius);
% ylim([0,100]);
% subplot(2,1,2)
% plot(sn_Curvature);

%%

figure(12)
plot(FC_Tot_Speed);
hold on;
plot(CurvMdlSpd);
hold on;
plot(Small_RadIdx,Spd_SmallRad);
hold off;
