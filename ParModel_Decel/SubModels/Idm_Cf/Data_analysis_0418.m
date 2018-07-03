clc; close all; clear all; ColorCode;
%% Load the simulation data
Simulation_data_set = load('Simulation_data_0418.mat');

start_brake_time_set = [];
start_relative_distance_set = [];
start_relative_velocity_set = [];
start_vehicle_velocity_set = [];
Max_brake_torque_set = [];

%% Collect the required data
for data_idx = 1:size(Simulation_data_set.Data,1)
% for data_idx = 21:21
    Simulation_data = Simulation_data_set.Data(data_idx);
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
    
    [Max_brake_torque, Max_brake_idx] = max(Brake_torque_FL);
    Max_brake_time = Time(Max_brake_idx);
    
    Brake_start_idx = [];
    for time_idx = 2:size(Time,2)
        if ((Brake_torque_FL(time_idx-1) == 0) && (Brake_torque_FL(time_idx) ~= 0))
            Brake_start_idx = [Brake_start_idx; time_idx];
            Max_filter = Brake_start_idx - 1501;
%             Max_filter = Brake_start_idx - Max_brake_idx;
            [Filter_val, Filter_idx] = min(abs(Max_filter));
            Brake_start_idx_filtered = Brake_start_idx(Filter_idx);
            Brake_start_time = Time(Brake_start_idx_filtered);
        end
    end
    
    start_relative_distance = Radar_relative_distance(Brake_start_idx_filtered);
    start_relative_velocity = Radar_relative_velocity(Brake_start_idx_filtered);
    start_vehicle_velocity = Car_v(Brake_start_idx_filtered);
    start_brake_time_set = [start_brake_time_set; Brake_start_time];
    start_relative_distance_set = [start_relative_distance_set; start_relative_distance];
    start_relative_velocity_set = [start_relative_velocity_set; start_relative_velocity];
    start_vehicle_velocity_set = [start_vehicle_velocity_set; start_vehicle_velocity];
    Max_brake_torque_set = [Max_brake_torque_set; Max_brake_torque];
end
%% Anaylize the relationship between the data
figure;
plot(start_vehicle_velocity_set,start_brake_time_set-30,'o','Color',CP(5,:))
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Ego vehicle velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Brake start time [s]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

figure;
plot(start_relative_distance_set,start_brake_time_set-30,'o','Color',CP(5,:))
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Relative distance [m]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Brake start time [s]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

figure;
plot(start_relative_velocity_set,start_brake_time_set-30,'o','Color',CP(5,:))
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Relative velocity [m/s]','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Brake start time [s]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;

a = start_relative_distance_set;
b = abs(start_relative_velocity_set);
c = start_vehicle_velocity_set;
d = Max_brake_torque_set;

IDM = b.^4 + (2 + c*4 + c.*b/2/sqrt(10*5));

figure;
plot(b./c.^2,'o','Color',CP(5,:))
set(gca,'fontsize',12)
title('Correlation analysis','fontsize',16,'fontname','arial','fontweight','bold')
xlabel('Prameter','fontsize',14,'fontname','arial','fontweight','bold')
ylabel('Max brake torque [Nm]','fontsize',14,'fontname','arial','fontweight','bold')
grid on;
