clear all;
close all;

%%
% Kyuhwan
% load('../004_Data/Amsa_RdlMdl4_YK.mat') % RoadModel
% 
% Dataset = load('../005_Amsa/Amsa_Curvature4_YK.mat');
% 
% load('../004_Data/Midan_RdlMdl.mat') % RoadModel
% 
% Dataset = load('../006_Midan/Logging_M01.mat');
load('../004_Data/MidanA_RdlMdl03.mat') % RoadModel

Dataset = load('../006_Midan/Curvature_MidanA/Logging_M03_1axis.mat');


Totallength = (PPS_RoadSegment.breaks(end));

%% Data Preprocessing (Slicing)
Totallength = round(Totallength/10);
Totallength = 10*Totallength;
limit = Totallength; % 5100m (without traffic right), 7300m (Whole Track)
Sampling = 1; %10m


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

%% Flatten coordinate (Flatten Earth)
Lat = Dataset.Latitude(:,end);
Long = Dataset.Longitude(:,end);
Height = Dataset.Height(:,end);
XYZ_flat = [];
for i  = 1: length(Lat)
    XYZ_flat_temp = lla2flat([Lat(i), Long(i), Height(i)], [Lat(1),Long(1)], 0,0);
    XYZ_flat = [XYZ_flat; XYZ_flat_temp];
end

figure(99)
plot(XYZ_flat(:,1),XYZ_flat(:,2));
X_flat = XYZ_flat(:,1);
Y_flat = XYZ_flat(:,2);
LineTemp = linspace(1,length(X_flat),length(X_flat))';
Lines=[(1:size(X_flat,1))' (2:size(X_flat,1)+1)']; Lines(end,2)=1;
% Lines = [LineTemp(1:end,1),LineTemp(2:end,1)+1];
Curvature_flat = LineCurvature2D([X_flat,Y_flat],Lines);
axis equal;

dx  = gradient(X_flat);
ddx = gradient(dx);
dy  = gradient(Y_flat);
ddy = gradient(dy);
num   = dx .* ddy - ddx .* dy;
denom = dx .* dx + dy .* dy;
denom = sqrt(denom);
denom = denom .* denom .* denom;
curvature_flat = num ./ denom;
curvature_flat(denom < 0) = NaN;
%% Find XYZ coordinate

XYZ= single(fnval(B_SplineRoadModel,Location));
X = XYZ(1,:);
Y = XYZ(2,:);
Z = XYZ(3,:);
%% Curvature Parametric 


dB_SplineRoadModel = fnder(B_SplineRoadModel);
dVal_B_SplineRoadModel = single(fnval(dB_SplineRoadModel,Location));
ddB_SplineRoadModel = fnder(dB_SplineRoadModel);
ddVal_B_SplineRoadModel = single(fnval(ddB_SplineRoadModel,Location));
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
ThreeDim_CurvatureRoadModel = numCur ./ denCur;



TwoDim_numCur = sqrt([1 1 1]*abs(TwoDim_crossD_DD.*TwoDim_crossD_DD)'); % Numerator of curvature
TwoDim_denCur = sqrt([1 1 1]*abs(TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel)); % Denominator of curvature
TwoDim_CurvatureRoadModel = TwoDim_numCur ./ TwoDim_denCur;


%%
K=Cal_Curvature_TwoDim(B_SplineRoadModel,sn_Location)


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
for j = 1 : length(Roadway)
    CurPos  = Roadway(j);
    if CurPos<limit
        if CurPos - PrePos >=Sampling
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



flt_curvature = movmean(sn_Curvature,5);
%%
figure(1)
subplot(2,1,1)
plot(flt_curvature,'b','linewidth',4);
grid on;
subplot(2,1,2)
plot(flt_curvature,'r','linewidth',4);
grid on;
%% Plot map
figure(2)
plot(sn_X,sn_Y);
axis equal;

%% Find High curvature
CurveStandard = 0.009;
HighCurvature=find(flt_curvature>CurveStandard);

%% Plot map
figure(3)
plot(sn_X,sn_Y,'b','linewidth',4);
hold on;
plot(sn_X(HighCurvature),sn_Y(HighCurvature),'go','linewidth',2);


%%
HC_Curvature=flt_curvature;
HC_Curvature(flt_curvature<CurveStandard)=NaN;
HC_sn_Ego_Velocity = sn_Ego_Velocity;
HC_sn_Ego_Velocity(flt_curvature<CurveStandard)=NaN;
figure(4)
subplot(2,1,1)
plot(flt_curvature,'b','linewidth',4);
hold on;
plot(HC_Curvature,'go','linewidth',2);
subplot(2,1,2)
plot(sn_Ego_Velocity,'b','linewidth',4);
hold on;
plot(HC_sn_Ego_Velocity,'go','linewidth',2);

%%
CurveInit = 642;
CurveEnd = 692;
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
subplot(3,1,2)
plot(FC_Mid_Speed);
hold on;
plot(fun2(k_mid,FC_Mid_Curvature));
legend('true','Curvature Model');
subplot(3,1,3)
plot(FC_Exit_Speed);
hold on;
plot(fun3(k_exit,FC_Exit_Curvature));
legend('true','Curvature Model');
%%
CurvMdlSpd = [CurMdlSpd_Ent CurMdlSpd_Mid CurMdlSpd_Exit];
figure(6)
plot(FC_Tot_Speed);
hold on;
plot(CurvMdlSpd);
hold on;



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
plot(fun1(k_ent,FC_Ent_Curvature));
legend('true','Curvature Model');
subplot(2,1,2)
plot(FC_Exit_Speed);
hold on;
plot(fun3(k_exit,FC_Exit_Curvature));
legend('true','Curvature Model');
%%
CurvMdlSpd = [CurMdlSpd_Ent CurMdlSpd_Exit];
figure(8)
plot(FC_Tot_Speed);
hold on;
plot(CurvMdlSpd);
hold on;


%%
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
%%
FC_Radius = Radius(FirstCurve_Total+1);
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
