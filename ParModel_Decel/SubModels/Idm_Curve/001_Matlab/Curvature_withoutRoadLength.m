clear all;
close all;
%Kyuhwan
load('../004_Data/Amsa_RdlMdl_YK.mat') % RoadModel

Dataset = load('../005_Amsa/Amsa_Curvature_YK.mat');


Resize = 10; % 10m

Totallength = (PPS_RoadSegment.breaks(end));

Sampling = 0.01;
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

for RoadPath = 0 : Sampling: Totallength-Sampling
    
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

    Kappa(count) = abs(X_d(count)*Y_dd(count)-Y_d(count)*X_dd(count))/(X_d(count)^2+Y_d(count)^2)^(3/2);
    if RoadPath > PPS_RoadSegment.breaks(i+1)
        i = i+1;
    end
    count = count +1;
end


%%
HighKappa=find(Kappa>0.005);


%%
figure(1)
plot3(X,Y,Z,'r', 'linewidth', 3);
grid on;
figure(2)
plot(X,Y,'r', 'linewidth', 3);
hold on;
plot(X(HighKappa),Y(HighKappa),'o', 'linewidth', 3);
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
RoadLength = Dataset.RoadLength;


KappaPerRoad = InterPolatedData(Dataset.RoadLength, Kappa);
VelocityPerRoad= InterPolatedData(Dataset.RoadLength, Ego_Velocity);


%%
figure(3)
subplot(2,1,1)
plot(RoadLength,KappaPerRoad,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(2,1,2)
plot(RoadLength,VelocityPerRoad,'r','linewidth',4);

grid on;