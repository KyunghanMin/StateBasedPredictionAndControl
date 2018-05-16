clc; close all; clear all; ColorCode;
%% Load the simulation data
Simulation_data = load('Simulation_data_0414.mat');
Simulation_data = Simulation_data.Data;
%% Type the simulation condition
Speed_variation = zeros(5,4);
Speed_variation(1,:) = [50, 70, 50, 70];
Speed_variation(2,:) = [40, 70, 40, 70];
Speed_variation(3,:) = [50, 80, 50, 80];
Speed_variation(4,:) = [40, 80, 40, 80];
Speed_variation(5,:) = [60, 80, 60, 80];

%% Extract the vehicle's velocity
Car_v_exp1 = Simulation_data(1,:).Car_v.data;
Car_v_exp2 = Simulation_data(2,:).Car_v.data;
Car_v_exp3 = Simulation_data(3,:).Car_v.data;
Car_v_exp4 = Simulation_data(4,:).Car_v.data;
Car_v_exp5 = Simulation_data(5,:).Car_v.data;
Car_v_unit = Simulation_data(1,:).Car_v.unit;

%% Extract the vehicle's longitudinal acceleration
Car_ax_exp1 = Simulation_data(1,:).Car_ax.data;
Car_ax_exp2 = Simulation_data(2,:).Car_ax.data;
Car_ax_exp3 = Simulation_data(3,:).Car_ax.data;
Car_ax_exp4 = Simulation_data(4,:).Car_ax.data;
Car_ax_exp5 = Simulation_data(5,:).Car_ax.data;

%% Extract the preceding vehicle's longitudinal acceleration
Traffic_ax_exp1 = Simulation_data(1,:).Traffic_T00_a_0_x.data;
Traffic_ax_exp2 = Simulation_data(2,:).Traffic_T00_a_0_x.data;
Traffic_ax_exp3 = Simulation_data(3,:).Traffic_T00_a_0_x.data;
Traffic_ax_exp4 = Simulation_data(4,:).Traffic_T00_a_0_x.data;
Traffic_ax_exp5 = Simulation_data(5,:).Traffic_T00_a_0_x.data;

%% Extract the preceding vehicle's velocity
Traffic_v_exp1 = Simulation_data(1,:).Traffic_T00_v_0_x.data;
Traffic_v_exp2 = Simulation_data(2,:).Traffic_T00_v_0_x.data;
Traffic_v_exp3 = Simulation_data(3,:).Traffic_T00_v_0_x.data;
Traffic_v_exp4 = Simulation_data(4,:).Traffic_T00_v_0_x.data;
Traffic_v_exp5 = Simulation_data(5,:).Traffic_T00_v_0_x.data;

%% Extract time
Time_exp1 = Simulation_data(1,:).Time.data;
Time_exp2 = Simulation_data(2,:).Time.data;
Time_exp3 = Simulation_data(3,:).Time.data;
Time_exp4 = Simulation_data(4,:).Time.data;
Time_exp5 = Simulation_data(5,:).Time.data;

%% Plot the velocity data
figure;
plot(Time_exp5,Traffic_v_exp5,'Color',RP(8,:),'linewidth',3,'DisplayName','Preceding vehicle')
hold on;
plot(Time_exp5,Car_v_exp5,'Color',GP(5,:),'linewidth',3,'DisplayName','Following vehicle')
set(gca,'fontsize',12)
title('Velocity of the vehicle','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

%% Plot the acceleration data
figure;
plot(Time_exp4,Traffic_ax_exp4,'Color',RP(8,:),'linewidth',3,'DisplayName','Preceding vehicle')
hold on;
plot(Time_exp4,Car_ax_exp4,'Color',GP(5,:),'linewidth',3,'DisplayName','Following vehicle')
set(gca,'fontsize',12)
title('Acceleration of the vehicle','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('accleration [m/s^2]','fontsize',14,'fontname','arial','fontweight','bold')
legend('show')
grid on;

