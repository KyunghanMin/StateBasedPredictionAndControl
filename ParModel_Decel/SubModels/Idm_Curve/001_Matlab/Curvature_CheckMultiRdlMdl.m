clear all;
close all;
%%
load('../004_Data/Kappa/Amsa_Kappa.mat')
RoadLength1= RoadLength;
KappaPerRoad1=KappaPerRoad;
VelocityPerRoad1=VelocityPerRoad;
%%
load('../004_Data/Kappa/Amsa_Kappa2.mat')
RoadLength2= RoadLength;
KappaPerRoad2=KappaPerRoad;
VelocityPerRoad2=VelocityPerRoad;
%%
load('../004_Data/Kappa/Amsa_Kappa3.mat')
RoadLength3= RoadLength;
KappaPerRoad3=KappaPerRoad;
VelocityPerRoad3=VelocityPerRoad;
%%
load('../004_Data/Kappa/Amsa_Kappa4.mat')
RoadLength4= RoadLength;
KappaPerRoad4=KappaPerRoad;
VelocityPerRoad4=VelocityPerRoad;
%%
load('../004_Data/Kappa/Amsa_Kappa_YK.mat')
RoadLength5= RoadLength;
KappaPerRoad5=KappaPerRoad;
VelocityPerRoad5=VelocityPerRoad;
%%
load('../004_Data/Kappa/Amsa_Kappa2_YK.mat')
RoadLength6= RoadLength;
KappaPerRoad6=KappaPerRoad;
VelocityPerRoad6=VelocityPerRoad;
%%
load('../004_Data/Kappa/Amsa_Kappa3_YK.mat')
RoadLength7= RoadLength;
KappaPerRoad7=KappaPerRoad;
VelocityPerRoad7=VelocityPerRoad;



%%
load('../004_Data/Kappa/Amsa_Kappa4_YK.mat')
RoadLength1= RoadLength;
KappaPerRoad1=KappaPerRoad;
VelocityPerRoad1=VelocityPerRoad;



%%
figure(3)
subplot(4,2,1)
plot(RoadLength1,KappaPerRoad1,'b','linewidth',4);
ylim([0,0.1]);
grid on;
subplot(4,2,2)
plot(RoadLength1,VelocityPerRoad1,'r','linewidth',4);
grid on;
subplot(4,2,3)
plot(RoadLength2,KappaPerRoad2,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(4,2,4)
plot(RoadLength2,VelocityPerRoad2,'r','linewidth',4);
grid on;
subplot(4,2,5)
plot(RoadLength3,KappaPerRoad3,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(4,2,6)
plot(RoadLength3,VelocityPerRoad3,'r','linewidth',4);
grid on;
subplot(4,2,7)
plot(RoadLength4,KappaPerRoad4,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(4,2,8)
plot(RoadLength4,VelocityPerRoad4,'r','linewidth',4);
grid on;



%%
figure(4)
subplot(4,2,1)
plot(RoadLength5,KappaPerRoad5,'b','linewidth',4);
ylim([0,0.1]);
grid on;
subplot(4,2,2)
plot(RoadLength5,VelocityPerRoad5,'r','linewidth',4);
grid on;
subplot(4,2,3)
plot(RoadLength6,KappaPerRoad6,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(4,2,4)
plot(RoadLength6,VelocityPerRoad6,'r','linewidth',4);
grid on;
subplot(4,2,5)
plot(RoadLength7,KappaPerRoad7,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(4,2,6)
plot(RoadLength7,VelocityPerRoad7,'r','linewidth',4);
grid on;
subplot(4,2,7)
plot(RoadLength,KappaPerRoad,'b','linewidth',4);
grid on;
ylim([0,0.1]);
subplot(4,2,8)
plot(RoadLength,VelocityPerRoad,'r','linewidth',4);
grid on;
