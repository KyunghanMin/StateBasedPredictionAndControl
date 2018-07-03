clc; close all; clear all; ColorCode;
%% Open the cmenv
cd ('C:\CM_Projects\Deceleartion_prediction\src_cm4sl')
cmenv;

%% Load the simulation data
cd('C:\CM_Projects\Deceleartion_prediction\SimOutput\Gyubin-PC\20180417')
Simulation_data = cmread('Simulation_1_211352.erg');
cd('C:\Users\Gyubin\OneDrive\¹®¼­\ACE Lab\GRG\2018 CX\Deceleration_prediction_car_following\CarMaker_simulation')

%% Process the simulation data set
Brake_pedal = Simulation_data.Brake_Pedal.data;
Brake_pedal_unit = Simulation_data.Brake_Pedal.unit;

Regenerative_torque_FL = Simulation_data.Brake_Trq_Reg_trg_FL.data;
Regenerative_torque_FR = Simulation_data.Brake_Trq_Reg_trg_FR.data;
Regenerative_torque_RL = Simulation_data.Brake_Trq_Reg_trg_RL.data;
Regenerative_torque_RR = Simulation_data.Brake_Trq_Reg_trg_RR.data;
Regernerative_torque_unit = Simulation_data.Brake_Trq_Reg_trg_FL.unit;

Brake_torque_FL = Simulation_data.Brake_Trq_WB_FL.data;
Brake_torque_FR = Simulation_data.Brake_Trq_WB_FR.data;
Brake_torque_RL = Simulation_data.Brake_Trq_WB_RL.data;
Brake_torque_RR = Simulation_data.Brake_Trq_WB_RR.data;
Brake_torque_unit = Simulation_data.Brake_Trq_WB_RR.unit;

Car_ax = Simulation_data.Car_ax.data;
Car_ax_unit = Simulation_data.Car_ax.unit;

Car_v = Simulation_data.Car_v.data;
Car_v_unit = Simulation_data.Car_v.unit;

Driver_brake = Simulation_data.Driver_Brake.data;
Driver_brake_unit = Simulation_data.Driver_Brake.unit;

Obj_relative_distance = Simulation_data.Sensor_Radar_RA00_Obj0_DistX.data;
Obj_relative_distance_unit = Simulation_data.Sensor_Radar_RA00_Obj0_DistX.unit;

Obj_relative_velocity = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.data;
Obj_relative_velocity_neg = -Obj_relative_velocity;
Obj_relative_velocity_unit = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.unit;

Radar_relative_distance = Simulation_data.Sensor_Radar_RA00_Obj0_DistX.data;
Radar_relative_distance_unit = Simulation_data.Sensor_Radar_RA00_Obj0_DistX.unit;

Radar_relative_velocity = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.data;
Radar_relative_velocity_neg = - Radar_relative_velocity;
Radar_relative_velocity_unit = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.unit;


Time = Simulation_data.Time.data;
Time_unit = Simulation_data.Time.unit;

Preceding_a = Simulation_data.Traffic_T00_a_0_x.data;
Preceding_a_unit = Simulation_data.Traffic_T00_a_0_x.unit;

Preceding_v = Simulation_data.Traffic_T00_v_0_x.data;
Preceding_v_unit = Simulation_data.Traffic_T00_v_0_x.unit;

Obj_relative_velocity_n = Obj_relative_velocity(Obj_relative_velocity < 0);
Brake_torque_obj_n = Brake_torque_FL(Obj_relative_velocity < 0);
Radar_relative_velocity_n = Radar_relative_velocity(Radar_relative_velocity < 0);
Brake_torque_radar_n = Brake_torque_FL(Radar_relative_velocity < 0);

%% Anaylize the relationship between the data
% figure;
% plot(Obj_relative_distance,Brake_torque_FL,'o','Color',CP(28,:))
% set(gca,'fontsize',12)
% title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
% xlabel('Relative distance [m]','fontsize',14,'fontname','arial','fontweight','bold')
% ylabel('Brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
% grid on;
% 
% figure;
% plot(Obj_relative_velocity,Brake_torque_FL,'o','Color',CP(28,:))
% set(gca,'fontsize',12)
% title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
% xlabel('Relative velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
% ylabel('Brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
% grid on;
% 
% figure;
% plot(Obj_relative_velocity_n,Brake_torque_obj_n,'o','Color',CP(28,:))
% set(gca,'fontsize',12)
% title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
% xlabel('Relative velocity (Obj, -) [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
% ylabel('Brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
% grid on;

% figure;
% plot(Radar_relative_velocity_n,Brake_torque_radar_n,'o','Color',CP(28,:))
% set(gca,'fontsize',12)
% title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
% xlabel('Relative velocity (Radar, -) [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
% ylabel('Brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
% grid on;

figure;
plot(Time,-Obj_relative_velocity,'Color',CP(14,:),'linewidth',3,'DisplayName','Relative velocity')
hold on;
plot(Time,Brake_torque_FL/70,'Color',CP(28,:),'linewidth',3,'DisplayName','Brake torque')
hold on;
plot(Time,Obj_relative_distance/20,'--','Color',CP(5,:),'linewidth',3,'DisplayName','Relative distance')
set(gca,'fontsize',12)
hold on;
plot(Time,Car_v/5,'Color',CP(26,:),'linewidth',3,'DisplayName','Vehicle speed')
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
% ylabel('Brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
subplot(3,2,1)
plot(Time,Brake_torque_FL,'Color',CP(14,:),'linewidth',3)
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

% figure;
subplot(3,2,2)
plot(Time,Radar_relative_distance,'Color',CP(26,:),'linewidth',3)
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Relative distance [m]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

% figure;
subplot(3,2,3)
plot(Time,Radar_relative_velocity,'Color',CP(5,:),'linewidth',3)
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Relative velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

% figure;
subplot(3,2,4)
plot(Time,Car_v,'Color',CP(28,:),'linewidth',3)
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Ego vehicle velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

subplot(3,2,5)
plot(Time,Car_ax,'Color',CP(28,:),'linewidth',3)
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Ego vehicle acceleration [m/s^2]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

figure;
plot(Time,Car_ax,'Color',CP(28,:),'linewidth',3)
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Ego vehicle acceleration [m/s^2]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;