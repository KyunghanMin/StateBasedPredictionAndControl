clear all;
close all;
%%
% Kyuhwan
load('../004_Data/Amsa_RdlMdl3_YK.mat') % RoadModel

Dataset = load('../005_Amsa/Amsa_Curvature3_YK.mat');

% load('../004_Data/Midan_RdlMdl.mat') % RoadModel
% 
% Dataset = load('../006_Midan/Logging_M01.mat');
% 
% 

Resize = 10; % 10m

Totallength = (PPS_RoadSegment.breaks(end));

Sampling = 1;
X = zeros(round(Totallength/Sampling),1);
Y = zeros(round(Totallength/Sampling),1);
Z = zeros(round(Totallength/Sampling),1);

X_d = zeros(round(Totallength/Sampling),1);
Y_d = zeros(round(Totallength/Sampling),1);
Z_d = zeros(round(Totallength/Sampling),1);

X_dd = zeros(round(Totallength/Sampling),1);
Y_dd = zeros(round(Totallength/Sampling),1);
Z_dd = zeros(round(Totallength/Sampling),1);

Kappa = zeros(round(Totallength/Sampling),1);

%%

i=1;
count = 1;
RoadPatharr = zeros(0,round((Totallength-Sampling)/Sampling));
%%

for RoadPath = 0 : Sampling: Totallength
    
    X(count) = PPS_RoadSegment.coefs(3*(i-1)+1,1)*(RoadPath-PPS_RoadSegment.breaks(i))^3+...
        PPS_RoadSegment.coefs(3*(i-1)+1,2)*(RoadPath-PPS_RoadSegment.breaks(i))^2+...
        PPS_RoadSegment.coefs(3*(i-1)+1,3)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        PPS_RoadSegment.coefs(3*(i-1)+1,4)*(RoadPath-PPS_RoadSegment.breaks(i))^0;
    Y(count) = PPS_RoadSegment.coefs(3*(i-1)+2,1)*(RoadPath-PPS_RoadSegment.breaks(i))^3+...
        PPS_RoadSegment.coefs(3*(i-1)+2,2)*(RoadPath-PPS_RoadSegment.breaks(i))^2+...
        PPS_RoadSegment.coefs(3*(i-1)+2,3)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        PPS_RoadSegment.coefs(3*(i-1)+2,4)*(RoadPath-PPS_RoadSegment.breaks(i))^0;    
    Z(count) = PPS_RoadSegment.coefs(3*(i-1)+3,1)*(RoadPath-PPS_RoadSegment.breaks(i))^3+...
        PPS_RoadSegment.coefs(3*(i-1)+3,2)*(RoadPath-PPS_RoadSegment.breaks(i))^2+...
        PPS_RoadSegment.coefs(3*(i-1)+3,3)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        PPS_RoadSegment.coefs(3*(i-1)+3,4)*(RoadPath-PPS_RoadSegment.breaks(i))^0;

    X_d(count) = 3*PPS_RoadSegment.coefs(3*(i-1)+1,1)*(RoadPath-PPS_RoadSegment.breaks(i))^2+...
        2*PPS_RoadSegment.coefs(3*(i-1)+1,2)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        PPS_RoadSegment.coefs(3*(i-1)+1,3)*(RoadPath-PPS_RoadSegment.breaks(i))^0;
    Y_d(count) = 3*PPS_RoadSegment.coefs(3*(i-1)+2,1)*(RoadPath-PPS_RoadSegment.breaks(i))^2+...
        2*PPS_RoadSegment.coefs(3*(i-1)+2,2)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        PPS_RoadSegment.coefs(3*(i-1)+2,3)*(RoadPath-PPS_RoadSegment.breaks(i))^0;
    Z_d(count) = 3*PPS_RoadSegment.coefs(3*(i-1)+3,1)*(RoadPath-PPS_RoadSegment.breaks(i))^2+...
        2*PPS_RoadSegment.coefs(3*(i-1)+3,2)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        PPS_RoadSegment.coefs(3*(i-1)+3,3)*(RoadPath-PPS_RoadSegment.breaks(i))^0;

    X_dd(count) = 6*PPS_RoadSegment.coefs(3*(i-1)+1,1)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        2*PPS_RoadSegment.coefs(3*(i-1)+1,2)*(RoadPath-PPS_RoadSegment.breaks(i))^0;
    Y_dd(count) = 6*PPS_RoadSegment.coefs(3*(i-1)+2,1)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        2*PPS_RoadSegment.coefs(3*(i-1)+2,2)*(RoadPath-PPS_RoadSegment.breaks(i))^0;
    Z_dd(count) = 6*PPS_RoadSegment.coefs(3*(i-1)+3,1)*(RoadPath-PPS_RoadSegment.breaks(i))^1+...
        2*PPS_RoadSegment.coefs(3*(i-1)+3,2)*(RoadPath-PPS_RoadSegment.breaks(i))^0;

    Kappa(count) = abs((X_d(count)*Y_dd(count))-(Y_d(count)*X_dd(count)))/(X_d(count)^2+Y_d(count)^2)^(3/2);
    if RoadPath > PPS_RoadSegment.breaks(i+1)
        i = i+1;
    end
    RoadPatharr(count)=RoadPath;
    count = count +1;
    

end


%%
HighKappa=find(Kappa>0.03);


%%
figure(1)
plot3(X,Y,Z,'r', 'linewidth', 3);
grid on;
figure(2)
plot(X,Y,'r', 'linewidth', 3);
hold on;
plot(X(HighKappa),Y(HighKappa),'go' ,'linewidth', 3);
grid on;
% for i = 1:length(X)
%     if Kappa(i)>0.03
%         scatter(X(i),Y(i),'g');
%     
%     end
% 
% end





%% Correlation
Ego_Velocity = (Dataset.WHL_SPD_RR+Dataset.WHL_SPD_RL+Dataset.WHL_SPD_FR+Dataset.WHL_SPD_FL)/4;    
Speed_Interp=interp1( linspace(0,1,numel(Ego_Velocity)), Ego_Velocity, linspace(0,1,length(Kappa) ));
Correlation_Speed = corrcoef(Kappa',Speed_Interp);
Long_Accel = Dataset.LONG_ACCEL;
Long_Accel_Interp=interp1( linspace(0,1,numel(Long_Accel)), Long_Accel, linspace(0,1,length(Kappa) ));
Correlation_Long_Accel = corrcoef(Kappa',Long_Accel_Interp);

Lat_Accel = Dataset.LAT_ACCEL;
Lat_Accel_Interp=interp1( linspace(0,1,numel(Lat_Accel)), Lat_Accel, linspace(0,1,length(Kappa) ));
Correlation_Lat_Accel = corrcoef(Kappa',Lat_Accel_Interp);

Brake_Pres = Dataset.CYL_PRES;
Brake_Pres_Interp=interp1( linspace(0,1,numel(Brake_Pres)), Brake_Pres, linspace(0,1,length(Kappa) ));
Correlation_Brake_Pres = corrcoef(Kappa',Brake_Pres_Interp);
%% Interpolation  to sn coordinate
RoadLength = RoadPatharr;


KappaPerRoad = InterPolatedData(RoadLength, Kappa);
VelocityPerRoad= InterPolatedData(RoadLength, Ego_Velocity);
LatAccPerRoad = InterPolatedData(RoadLength, Lat_Accel);


%%
HighKappaPerRoad = -ones(1,length(KappaPerRoad));
for i = 1:1: round(length(KappaPerRoad))
    if KappaPerRoad(i)>0.03
        HighKappaPerRoad(i) = KappaPerRoad(i);
    end
end

    
%%
figure(3)
subplot(2,1,1)



plot(RoadLength,KappaPerRoad,'b','linewidth',4);
hold on;
plot(RoadLength,HighKappaPerRoad,'go','linewidth',2);
ylim([0,0.1]);
grid on;
subplot(2,1,2)
plot(RoadLength,VelocityPerRoad,'r','linewidth',4);
grid on;


%% Curvature Parametric 


dB_SplineRoadModel = fnder(B_SplineRoadModel);
dVal_B_SplineRoadModel = fnval(dB_SplineRoadModel,RoadPatharr);
ddB_SplineRoadModel = fnder(dB_SplineRoadModel);
ddVal_B_SplineRoadModel = fnval(ddB_SplineRoadModel,RoadPatharr);
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




%%
figure(4)

subplot(2,1,1)
plot(RoadLength,KappaPerRoad,'b','linewidth',4);
hold on;
plot(RoadLength,CurvatureRoadModel,'r','linewidth',2);
hold on;
plot(RoadLength,TwoDim_CurvatureRoadModel,'g','linewidth',2); % 이걸 사용
grid on;
legend('로드모델 재구성해서 곡률구했을때','3dim 로드모델바로미분','2dim 로드모델바로미분');
subplot(2,1,2)
plot(RoadLength,VelocityPerRoad,'r','linewidth',4);
grid on;
%%

filtered_curvature = movmean(TwoDim_CurvatureRoadModel,5);
figure(98)

plot( filtered_curvature);

FirstCornerPos=find(RoadLength>680 & RoadLength<730);

figure(97)
plot(VelocityPerRoad(FirstCornerPos))


figure(96)
plot(X,Y,'r', 'linewidth', 3);
hold on;
plot(X(FirstCornerPos),Y(FirstCornerPos),'go' ,'linewidth', 3);
grid on;