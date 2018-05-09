clc; close all; clear all; ColorCode;

%% Load the simulation data
Simulation_data_set = load('Simulation_data_0418.mat');

%% Collect the required data
max_brake_torque = [];
min_relative_distance = [];
min_relative_velocity = [];

Simulation_data = Simulation_data_set.Data(21,:);
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

time = [0, 60, 120, 180];

Brake_torque_section = [];
Car_v_section = [];
Obj_relative_distance_section = [];
Obj_relative_velocity_section = [];
Time_section = [];

for time_idx = 1:(size(time,2)-1)
    Brake_torque_FL_temp = Brake_torque_FL((Time>=time(time_idx))&(Time<time(time_idx+1)));
    Car_v_temp = Car_v((Time>=time(time_idx))&(Time<time(time_idx+1)));
    Obj_relative_distance_temp = Obj_relative_distance((Time>=time(time_idx))&(Time<time(time_idx+1)));
    Obj_relative_velocity_temp = Obj_relative_velocity((Time>=time(time_idx))&(Time<time(time_idx+1)));
    Time_temp = Time((Time>=time(time_idx))&(Time<time(time_idx+1)));
    
    max_brake_torque = [max_brake_torque; max(Brake_torque_FL_temp)];
    min_relative_distance = [min_relative_distance; min(Obj_relative_distance_temp(Obj_relative_distance_temp>0))];
    min_relative_velocity = [min_relative_velocity; min(Obj_relative_velocity_temp)];
    
    Brake_torque_section = [Brake_torque_section; Brake_torque_FL_temp];
    Car_v_section = [Car_v_section; Car_v_temp];
    Obj_relative_distance_section = [Obj_relative_distance_section; Obj_relative_distance_temp];
    Obj_relative_velocity_section = [Obj_relative_velocity_section; Obj_relative_velocity_temp];
    Time_section = [Time_section; Time_temp];
    
end

%% Anaylize the relationship between the data
figure;
plot(Time,Preceding_v,'Color',CP(10,:),'DisplayName','Preceding velocity','linewidth',3)
title('The preceding vehicle velocity','fontsize',16,'fontname','arial','fontweight','bold','color','w')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
set(gca,'color','none')
set(gca,'xcolor','w')
set(gca,'ycolor','w')
set(gca,'fontweight','bold')
grid on;

figure;
plot(Time,Car_v,'Color',CP(16,:),'DisplayName','Ego velocity','linewidth',3)
title('The ego vehicle velocity','fontsize',16,'fontname','arial','fontweight','bold','color','w')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
set(gca,'color','none')
set(gca,'xcolor','w')
set(gca,'ycolor','w')
set(gca,'fontweight','bold')
grid on;

figure;
plot(Time,Car_ax,'Color',CP(26,:),'DisplayName','Ego acceleration','linewidth',3)
title('The ego vehicle acceleration','fontsize',16,'fontname','arial','fontweight','bold','color','w')
xlabel('Time [s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Acceleration [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
set(gca,'color','none')
set(gca,'xcolor','w')
set(gca,'ycolor','w')
set(gca,'fontweight','bold')
grid on;
