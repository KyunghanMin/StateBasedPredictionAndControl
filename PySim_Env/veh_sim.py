# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:42:50 2018

@author: Kyunghan
"""
#%% 0. Model import
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from veh_mod import Mod_Veh, Mod_Behavior, Mod_Driver, Mod_Body, Mod_PowerTrain, Mod_Env, type_DataLog
#%% 1. Single vehicle driven - longitudinal input set
# 1.1 Import models
# Powertrain import and configuration
kona_power = Mod_PowerTrain()
# ~~~~~
# Bodymodel import and configuration
kona_body = Mod_Body()
kona_body.Dyn_config(10)
# ~~~~
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_body)

# 1.2 Simulation config
Ts = 0.01
sim_time = 40
sim_time_range = np.arange(0,sim_time,0.01)

# ----------------------------- select input set ---------------------------
Input_index = 2
if Input_index == 0:
# Go straight : Input_index = 0
    u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 * np.ones(int(len(sim_time_range)*0.9))))
    u_brk_val = 0 * np.ones(len(sim_time_range))
elif Input_index == 1:    
# Sin wave : Input_index = 1
    u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 + 0.1*np.sin(0.01*np.arange(int(len(sim_time_range)*0.9)))))
    u_brk_val = 0 * np.ones(len(sim_time_range))
# Brake : Input_index = 2
elif Input_index == 2:
    u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 * np.ones(int(len(sim_time_range)*0.4)),  0 * np.ones(int(len(sim_time_range)*0.5))))
    u_brk_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.55)), 0.3 * np.ones(int(len(sim_time_range)*0.45))))    
else:
    print('입력을 똑바로 하세요 ~_~')
    
# 1.3 Simulation
# Set logging data
sim1 = type_DataLog(['Veh_Vel','Pos_X','Pos_Y','Acc_Set','Brk_Set'])
for sim_step in range(len(sim_time_range)):
    # Arrange vehicle input
    u_acc_in = u_acc_val[sim_step]
    u_brk_in = u_brk_val[sim_step]
    # Vehicle model sim
    [veh_vel, the_wheel] = kona_vehicle.Veh_driven(u_acc = u_acc_in, u_brake = u_brk_in)
    [pos_x, pos_y, pos_s, pos_n, psi_veh] = kona_vehicle.Veh_position_update(veh_vel, the_wheel)    
    # Store data
    sim1.StoreData([veh_vel, pos_x, pos_y, u_acc_in, u_brk_in])

[sim1_veh_vel, sim1_pos_x, sim1_pos_y, sim1_u_acc, sim1_u_brk] = sim1.get_profile_value(['Veh_Vel','Pos_X','Pos_Y','Acc_Set','Brk_Set'])

# 1.4 Reulst plot
fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)
ax1.plot(sim1_pos_x, sim1_pos_y);ax1.set_xlabel('X position [m]');ax1.set_ylabel('Y position [m]')
ax2.plot(sim_time_range, sim1_veh_vel)
ax3.plot(sim_time_range, sim1_u_acc,label='Acc')
ax3.plot(sim_time_range, sim1_u_brk,label='Brk')
ax3.legend()
#%% 2. Single vehicle driven - longitudinal input set
# 1.1 Import models
# Powertrain import and configuration
kona_power = Mod_PowerTrain()
# ~~~~~
# Bodymodel import and configuration
kona_body = Mod_Body()
# ~~~~
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_body)

# 1.2 Simulation config
Ts = 0.01
sim_time = 40
sim_time_range = np.arange(0,sim_time,0.01)
# ----------------------------- select input set ---------------------------
Input_index = 1
if Input_index == 0:
    # Turn right
    u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 * np.ones(int(len(sim_time_range)*0.9))))
    u_brk_val = 0 * np.ones(len(sim_time_range))
    u_steer_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.3)), 0.01 * np.ones(int(len(sim_time_range)*0.7))))
elif Input_index == 1:
    # Sin input
    u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 * np.ones(int(len(sim_time_range)*0.9))))
    u_brk_val = 0 * np.ones(len(sim_time_range))
    u_steer_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.3)), 0.01*np.sin(0.01*np.arange(int(len(sim_time_range)*0.7)))))
else:
    print('입력을 똑바로 하세요 ~_~')
# 1.3 Simulation
# Set logging data
sim2 = type_DataLog(['Veh_Vel','Pos_X','Pos_Y','Acc_Set','Str_Set'])
for sim_step in range(len(sim_time_range)):
    # Arrange vehicle input
    u_acc_in = u_acc_val[sim_step]
    u_brk_in = u_brk_val[sim_step]
    u_str_in = u_steer_val[sim_step]
    # Vehicle model sim
    [veh_vel, the_wheel] = kona_vehicle.Veh_driven(u_acc = u_acc_in, u_brake = u_brk_in, u_steer = u_str_in)
    [pos_x, pos_y, pos_s, pos_n, psi_veh] = kona_vehicle.Veh_position_update(veh_vel, the_wheel)    
    # Store data
    sim2.StoreData([veh_vel, pos_x, pos_y, u_acc_in, u_str_in])    

[sim2_veh_vel, sim2_pos_x, sim2_pos_y, sim2_u_acc, sim2_u_str] = sim2.get_profile_value(['Veh_Vel','Pos_X','Pos_Y','Acc_Set','Str_Set'])

# 1.4 Reulst plot
fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)
ax1.plot(sim1_pos_x, sim1_pos_y);ax1.set_xlabel('X position [m]');ax1.set_ylabel('Y position [m]');ax1.axis('equal')
ax2.plot(sim_time_range, sim2_veh_vel)
ax3.plot(sim_time_range, sim2_u_acc,label = 'Acc')
ax3.plot(sim_time_range, sim2_u_str,label = 'Str')
#%% 3. Driver driven - Speed control
# 3.1 Import models
# Powertrain import and configuration
kona_power = Mod_PowerTrain()
# ~~~~~
# Bodymodel import and configuration
kona_body = Mod_Body()
# ~~~~
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_body)
# ~~~~
# Driver model
drv_kyunghan = Mod_Driver()
# ~~~~
# Behavior model
beh_cruise = Mod_Behavior(drv_kyunghan)
# ~~~~

# 3.2 Simulation config
Ts = 0.01
sim_time = 70
sim_time_range = np.arange(0,sim_time,0.01)
veh_vel = 0 # Initial vehicle speed

# ----------------------------- select input set ---------------------------
Input_index = 3
if Input_index == 0:
    # Driver = normal
    drv_kyunghan.set_char('Normal')
    beh_cruise.Drver_set(drv_kyunghan)
elif Input_index == 1:
    # Driver = aggressive
    drv_kyunghan.set_char('Aggressive')
    beh_cruise.Drver_set(drv_kyunghan)
elif Input_index == 2:
    # Driver = defensive
    drv_kyunghan.set_char('Defensive')
    beh_cruise.Drver_set(drv_kyunghan)
elif Input_index == 3:
    # Driver = new driver with config parameter
    drv_kyuhwan = Mod_Driver()
    drv_kyuhwan.P_gain_lon = 1.2
    beh_cruise.Drver_set(drv_kyuhwan)
else:
    print('입력을 똑바로 하세요 ~_~')
    
u_speed_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 30 * np.ones(int(len(sim_time_range)*0.9))))

# 3.3 Simulation
# Set logging data
sim3 = type_DataLog(['Veh_Vel','Acc_in','Brk_in','Trq_set'])

for sim_step in range(len(sim_time_range)):
    # Arrange cruise input
    u_speed_set = u_speed_val[sim_step]
    # Behavior control
    [u_acc_in, u_brk_in] = beh_cruise.Lon_control(u_speed_set, veh_vel)
    # Vehicle model sim
    [veh_vel, the_wheel] = kona_vehicle.Veh_driven(u_acc = u_acc_in, u_brake = u_brk_in)
    # Store data
    sim3.StoreData([veh_vel, u_acc_in, u_brk_in, beh_cruise.trq_set_lon])

[sim3_veh_vel, sim3_u_acc, sim3_u_brk, sim3_trq_set] = sim3.get_profile_value(['Veh_Vel','Acc_in','Brk_in','Trq_set'])

# 3.4 Reulst plot
fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.plot(sim_time_range, u_speed_val)
ax1.plot(sim_time_range, sim3_veh_vel)
ax2.plot(sim_time_range, sim3_u_acc)
ax2.plot(sim_time_range, sim3_u_brk)
ax3.plot(sim_time_range, sim3_trq_set)
#%% 4. Driver driven - Steering control
# 4.1 Import models
# Powertrain import and configuration
kona_power = Mod_PowerTrain()
# ~~~~~
# Bodymodel import and configuration
kona_body = Mod_Body()
# ~~~~
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_body)
# ~~~~
# Driver model
drv_kyunghan = Mod_Driver()
# ~~~~
# Behavior model
beh_steer = Mod_Behavior(drv_kyunghan)
# ~~~~
# Import road data
RoadData = io.loadmat('road_data_straight.mat')
X_in = np.reshape(RoadData['sn_X'],len(RoadData['sn_X']))
Y_in = np.reshape(RoadData['sn_Y'],len(RoadData['sn_X']))
# Environment model
env_st = Mod_Env(X_in,Y_in)

# 4.2 Simulation config
Ts = 0.01
sim_time = 70
sim_time_range = np.arange(0,sim_time,0.01)
veh_vel = 0 # Initial vehicle speed
pos_x = env_st.road_x[0]
pos_y = env_st.road_y[0]
psi_veh = env_st.road_ang[0]
u_speed_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 30 * np.ones(int(len(sim_time_range)*0.9))))

# ----------------------------- select input set ---------------------------
#Input_index = 3
#if Input_index == 0:
#    # Driver = normal
#    drv_kyunghan.set_char('Normal')
#    beh_cruise.Drver_set(drv_kyunghan)
#elif Input_index == 1:
#    # Driver = aggressive
#    drv_kyunghan.set_char('Aggressive')
#    beh_cruise.Drver_set(drv_kyunghan)
#elif Input_index == 2:
#    # Driver = defensive
#    drv_kyunghan.set_char('Defensive')
#    beh_cruise.Drver_set(drv_kyunghan)
#elif Input_index == 3:
#    # Driver = new driver with config parameter
#    drv_kyuhwan = Mod_Driver()
#    drv_kyuhwan.P_gain_lon = 1.2
#    beh_cruise.Drver_set(drv_kyuhwan)
#else:
#    print('입력을 똑바로 하세요 ~_~')
    
# 4.3 Simulation
# Set logging data
sim3 = type_DataLog(['Veh_Vel','Acc_in','Brk_in','Trq_set'])

for sim_step in range(len(sim_time_range)):
    # Arrange behavior input
    u_speed_set = u_speed_val[sim_step]
    u_lane_off = u_speed_val[sim_step]
    # Arrange Env condition
    road_x = env_st.road_x[sim_step]
    road_y = env_st.road_y[sim_step]
    road_angle = env_st.road_ang[sim_step]
    # Behavior control    
    [u_acc_in, u_brk_in] = beh_steer.Lon_control(u_speed_set, veh_vel)
    u_steer_in = beh_steer.Lat_behavior(pos_x, pos_y, psi_veh, road_x, road_y, road_angle)

    # Vehicle model sim
    [veh_vel, the_wheel] = kona_vehicle.Veh_driven(u_acc = u_acc_in, u_brake = u_brk_in, u_steer = u_steer_in)
    [pos_x, pos_y, pos_s, pos_n, psi_veh] = kona_vehicle.Veh_position_update(veh_vel, the_wheel)

    # Behavior sim
    
    # Store data
    sim3.StoreData([veh_vel, u_acc_in, u_brk_in, beh_steer.trq_set_lon])

[sim3_veh_vel, sim3_u_acc, sim3_u_brk, sim3_trq_set] = sim3.get_profile_value(['Veh_Vel','Acc_in','Brk_in','Trq_set'])

# 4.4 Reulst plot
fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.plot(sim_time_range, u_speed_val)
ax1.plot(sim_time_range, sim3_veh_vel)
ax2.plot(sim_time_range, sim3_u_acc)
ax2.plot(sim_time_range, sim3_u_brk)
ax3.plot(sim_time_range, sim3_trq_set)
#%% Midan simulation
# Powertrain import and configuration
kona_power = Mod_PowerTrain()
# ~~~~~
# Bodymodel import and configuration
kona_body = Mod_Body()
# ~~~~
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_body)
# ~~~~
# Driver model
drv_kyunghan = Mod_Driver()
# ~~~~
# Behavior model
beh_steer = Mod_Behavior(drv_kyunghan)
# ~~~~
# Import road data
RoadData = io.loadmat('road_data_midan.mat')
X_in = np.reshape(RoadData['sn_X'],max(np.shape(RoadData['sn_X'])))
Y_in = np.reshape(RoadData['sn_Y'],max(np.shape(RoadData['sn_X'])))
# Environment model
env_st = Mod_Env(X_in,Y_in)
