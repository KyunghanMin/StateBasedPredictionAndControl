clc; close all; clear all; ColorCode;
%% Open the cmenv
cd ('C:\CM_Projects\Deceleartion_prediction\src_cm4sl')
cmenv;

%% Load the simulation data
cd('C:\CM_Projects\Deceleartion_prediction\SimOutput\Gyubin-PC\20180412')
Simulation_data = cmread('Simulation_1_090235.erg');

%% Save the Simulation_data
cd('C:\Users\Gyubin\OneDrive\¹®¼­\ACE Lab\GRG\2018 CX\Deceleration_prediction_car_following\CarMaker_simulation')
save Simulation_data.mat Simulation_data

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
Obj_relative_velocity_unit = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.unit;

Radar_relative_distance = Simulation_data.Sensor_Radar_RA00_Obj0_DistX.data;
Radar_relative_distance_unit = Simulation_data.Sensor_Radar_RA00_Obj0_DistX.unit;

Radar_relative_velocity = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.data;
Radar_relative_velocity_unit = Simulation_data.Sensor_Radar_RA00_Obj0_VrelX.unit;


Time = Simulation_data.Time.data;
Time_unit = Simulation_data.Time.unit;

Preceding_a = Simulation_data.Traffic_T00_a_0_x.data;
Preceding_a_unit = Simulation_data.Traffic_T00_a_0_x.unit;

Preceding_v = Simulation_data.Traffic_T00_v_0_x.data;
Preceding_v_unit = Simulation_data.Traffic_T00_v_0_x.unit;
%% Plot the simulation results
figure;
plot(Time,Car_v,'Color',CP(28,:),'linewidth',3,'DisplayName','The following')
hold on;
plot(Time,Preceding_v,'Color',CP(6,:),'linewidth',3,'DisplayName','The preceding')
set(gca,'fontsize',12)
title('Vehicle speed','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Speed [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
plot(Time,Car_ax,'Color',CP(28,:),'linewidth',3,'DisplayName','The following')
hold on;
plot(Time,Preceding_a,'Color',CP(6,:),'linewidth',3,'DisplayName','The preceding')
set(gca,'fontsize',12)
title('Vehicle acceleration','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Acceleration [m/s^2]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
plot(Time,Obj_relative_distance,'Color',CP(28,:),'linewidth',3,'DisplayName','Distance_r')
set(gca,'fontsize',12)
title('Obeject sensor','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Distance [m]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
plot(Time,Obj_relative_velocity,'Color',CP(28,:),'linewidth',3,'DisplayName','Velocity_r')
set(gca,'fontsize',12)
title('Obeject sensor','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
plot(Time,Brake_pedal,'Color',CP(28,:),'linewidth',3,'DisplayName','Brake pedal')
set(gca,'fontsize',12)
title('Brake pedal input','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Brake pedal','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
plot(Time,Regenerative_torque_FL,'Color',CP(28,:),'linewidth',3,'DisplayName','FL')
hold on;
plot(Time,Regenerative_torque_FR,'Color',CP(26,:),'linewidth',3,'DisplayName','FR')
hold on;
plot(Time,Regenerative_torque_RL,'Color',CP(16,:),'linewidth',3,'DisplayName','RL')
hold on;
plot(Time,Regenerative_torque_RR,'Color',CP(5,:),'linewidth',3,'DisplayName','RR')
set(gca,'fontsize',12)
title('Regenerative braking','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Regenerative braking torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

figure;
plot(Time,Brake_torque_FL,'Color',CP(28,:),'linewidth',3,'DisplayName','FL')
hold on;
plot(Time,Brake_torque_FR,'Color',CP(26,:),'linewidth',3,'DisplayName','FR')
hold on;
plot(Time,Brake_torque_RL,'Color',CP(16,:),'linewidth',3,'DisplayName','RL')
hold on;
plot(Time,Brake_torque_RR,'Color',CP(5,:),'linewidth',3,'DisplayName','RR')
set(gca,'fontsize',12)
title('Braking','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Braking torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

