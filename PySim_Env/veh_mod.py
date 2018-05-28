# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:44:05 2018

@author: Kyunghan
@vehicle simulation model
"""
from math import pi, sin, cos, sqrt, acos, atan
import numpy as np
from scipy.spatial import distance as dist_calc
import scipy.io as io
import matplotlib.pyplot as plt
# Simulation configuration
Ts = 0.01
#%% 0. local functions
#1. Calculation of radius
def Calc_Radius (x_in, y_in, filt_num = 0):
    if len(x_in) != len(y_in):
        print('Radius calculation error: Data length must be same!')
        return
    else:
        x_c_out = np.zeros(len(x_in))
        y_c_out = np.zeros(len(x_in))
        R_out = np.zeros(len(x_in))        
        circle_index = np.zeros(len(x_in))        
        mr_o = np.zeros(len(x_in))
        mt_o = np.zeros(len(x_in))
        for i in range(len(x_in)-2):
            x_lst = x_in[i:i+3]; y_lst = y_in[i:i+3]
            x1 = x_lst[0]; x2 = x_lst[1]; x3 = x_lst[2]; y1 = y_lst[0]; y2 = y_lst[1]; y3 = y_lst[2]    
            mr = (y2-y1)/(x2-x1);
            mt = (y3-y2)/(x3-x2);
            if np.equal(mr,mt) | np.isnan(mr) | np.isnan(mt):
                x_c = np.nan
                y_c = np.nan
                R = np.inf
                circle_on = 0
            else:            
                # Radius calculation
                if mr >= abs(100):
                    x_c = (mt*(y3-y1)+(x2+x3))/2       
                elif mt >= abs(100):
                    x_c = ((x1+x2) - mr*(y3-y1))/2
                else:
                    x_c = (mr*mt*(y3-y1)+mr*(x2+x3)-mt*(x1+x2))/(2*(mr-mt))            
                if mr == 0:
                    y_c = -1/mt*(x_c-(x2+x3)/2)+(y2+y3)/2
                else:
                    y_c = -1/mr*(x_c-(x1+x2)/2)+(y1+y2)/2
                R = np.sqrt(np.square((x_c-x1))+np.square((y_c-y1)))                
                circle_on = 1                
            x_c_out[i] = x_c
            y_c_out[i] = y_c
            R_out[i] = R
            circle_index[i] = circle_on
            mr_o[i] = mr
            mt_o[i] = mt            
        if filt_num !=0:
           R_out = Filt_MovAvg(R_out, filt_num)
        return [R_out, x_c_out, y_c_out, circle_index,mr_o,mt_o]

#2. Moving average filt
def Filt_MovAvg(Data, odd_filt_num):
    if len(Data) <= odd_filt_num:
        print('Moving average filter error: Data <= Filt Num')
        Data_FiltOut = 0
    elif odd_filt_num%2 == 0:
        print('Moving average filter error: Filt Num bust be odd number')
        Data_FiltOut = 0        
    else:
        tmpFiltNum_c = np.int(odd_filt_num/2)
        Data_FiltOut = np.zeros(len(Data))
        Data_FiltOut[0:tmpFiltNum_c] = Data[0:tmpFiltNum_c]
        for i in range(tmpFiltNum_c,len(Data),1):
            Data_FiltOut[i] = np.mean(Data[i-tmpFiltNum_c:i-tmpFiltNum_c+odd_filt_num])
    return Data_FiltOut

#3. Projection distance calculation
def Calc_PrDis(Data_Array_x, Data_Array_y, Current_point):
    dis_array = np.sqrt(np.square(Data_Array_x - Current_point[0]) + np.square(Data_Array_y - Current_point[1]))    
    min_index = np.argmin(dis_array)    
    tmp_c = dis_array[min_index]
    
    if dis_array[min_index-1] <= dis_array[min_index+1]:
        tmp_b = dis_array[min_index-1]
        tmp_a = dist_calc.euclidean([Data_Array_x[min_index], Data_Array_y[min_index]], [Data_Array_x[min_index-1], Data_Array_y[min_index-1]])
        road_an = atan((Data_Array_y[min_index] - Data_Array_y[min_index-1])/(Data_Array_x[min_index] - Data_Array_x[min_index-1]))
        veh_an = atan((Current_point[1]-Data_Array_y[min_index-1])/(Current_point[0]-Data_Array_x[min_index-1]))
    else:
        tmp_b = dis_array[min_index+1]
        tmp_a = dist_calc.euclidean([Data_Array_x[min_index], Data_Array_y[min_index]], [Data_Array_x[min_index+1], Data_Array_y[min_index+1]])
        road_an = atan((Data_Array_y[min_index+1] - Data_Array_y[min_index])/(Data_Array_x[min_index+1] - Data_Array_x[min_index]))
        veh_an = atan((Current_point[1]-Data_Array_y[min_index])/(Current_point[0]-Data_Array_x[min_index]))
    if (tmp_a**2 + tmp_c**2 - tmp_b**2)/(2*tmp_a*tmp_c) >= 1:
        tmp_dem = 1
    elif (tmp_a**2 + tmp_c**2 - tmp_b**2)/(2*tmp_a*tmp_c) <= -1:
        tmp_dem = -1
    else:
        tmp_dem = (tmp_a**2 + tmp_c**2 - tmp_b**2)/(2*tmp_a*tmp_c)
            
    if tmp_c == 0:
        tmp_An = 0
    else:
        tmp_An = acos(tmp_dem)
    tmp_n = sin(tmp_An)*tmp_c
    if veh_an >= road_an:
        tmp_dir = 'Right'
    else:
        tmp_dir = 'Left'            
    return tmp_n, tmp_dir
#%% 0. type definition

class type_pid_controller:
    def __init__(self, P_gain = 1, I_gain = 1, D_gain = 'None', Ts = 'None', I_val_old = 0, Error_old = 0):
        self.P_gain = P_gain
        self.I_gain = I_gain
        self.D_gain = D_gain        
        self.I_val_old = I_val_old
        self.Error_old = Error_old
        if Ts == 'None':
             self.Ts_loc = globals()['Ts']     
        else:
             self.Ts_loc = Ts        
    
    def Control(self,set_value,cur_value):
        self.Error = set_value - cur_value
        self.P_val = self.P_gain * self.Error
        self.I_val = self.I_gain * self.Error *self.Ts_loc + self.I_val_old
        self.I_val_old = self.I_val
        self.D_val = self.D_gain * (self.Error_old - self.Error)/self.Ts_loc
        Control_out = self.P_val + self.I_val + self.D_val
        return Control_out        

class type_objective:    
    def __init__(self):
        self.object_class = []
        self.object_param = []
        self.object_loc_s = []
    def add_object(self,object_class_in = 'None', object_param_in = 'None', object_loc_in = 0):
        self.object_class.append(object_class_in)
        self.object_param.append(object_param_in)
        self.object_loc_s.append(object_loc_in)
    def merg_object(self,object_class_in = 'None', object_param_in = 'None', object_loc_in = 0):
        self.object_class = self.object_class + object_class_in
        self.object_param = self.object_param + object_param_in
        self.object_loc_s = self.object_loc_s + object_loc_in
        
class type_drvstate:    
    def __init__(self):
        self.state = []
        self.state_param = []
        self.state_reldis = []
    def add_state(self, state_in = 'None', state_param_in = 'None', state_reldis_in = 0):
        self.state.append(state_in)
        self.state_param.append(state_param_in)
        self.state_reldis.append(state_reldis_in)
    def set_state(self, state_in = 'None', state_param_in = 'None', state_reldis_in = 0):
        self.state = state_in
        self.state_param = state_param_in
        self.state_reldis = state_reldis_in
#%% 1. Vehicle model
    # 1. Powertrain
    # 2. Body
    # 3. Vehicle (Powertrain, Body)
class Mod_PowerTrain:    
    def __init__(self):
        self.w_mot = 0
        self.t_mot = 0
        self.t_load = 0
        self.Motor_config()        
        self.DriveTrain_config()
        self.Ts_loc = globals()['Ts']        
        
    def Motor_config(self, conf_rm = 0.1, conf_lm = 0.1, conf_kb = 6.5e-4, conf_kt = 0.1, conf_jm = 1e-3, conf_trq_gain = 1):
        self.conf_rm_mot = conf_rm
        self.conf_lm_mot = conf_lm
        self.conf_kb_mot = conf_kb
        self.conf_kt_mot = conf_kt
        self.conf_jm_mot = conf_jm
        self.conf_trq_gain = conf_trq_gain
        
    def DriveTrain_config(self, conf_rd = 8, conf_ks = 0.01):        
        self.conf_rd_gear = conf_rd
        self.conf_ks_shaft = conf_ks
        
    def Motor_driven(self, v_in = 0, w_shaft = 0):
        # Elec motor model: Motor torque --> Mech motor model: Motor speed --> Drive shaft model: Load torque
        self.t_mot = self.Motor_elec_dynamics(self.t_mot, v_in, self.w_mot)
        self.w_mot = self.Motor_mech_dynamics(self.w_mot, self.t_mot, self.t_load)
        self.t_load = self.Drive_shaft_dynamics(self.t_load, self.w_mot, w_shaft)
        return [self.w_mot, self.t_mot, self.t_load]
    
    def Motor_control(self, t_mot_des = 0, w_shaft = 0):        
        v_in = self.Motor_torque_system(t_mot_des)
        self.Motor_driven(v_in, w_shaft)
        return [self.w_mot, self.t_mot, self.t_load]
    
    def Motor_elec_dynamics(self, t_mot, v_in, w_mot):
        # Motor torque calculation
        t_mot = t_mot*(1 - self.conf_rm_mot/self.conf_lm_mot * self.Ts_loc) \
        + self.Ts_loc*self.conf_kt_mot/self.conf_lm_mot * (v_in - self.conf_kb_mot * w_mot)
        return t_mot         
        
    def Motor_mech_dynamics(self, w_mot, t_mot, t_load):
        # Motor speed calculation
        w_mot =  w_mot + self.Ts_loc*(t_mot - t_load/self.conf_rd_gear)/self.conf_jm_mot
        return w_mot
        
    def Drive_shaft_dynamics(self, t_load, w_mot, w_shaft):
        t_load = t_load + self.Ts_loc*self.conf_ks_shaft*(w_mot/self.conf_rd_gear - w_shaft)
        return t_load
    
    def Motor_torque_system(self, t_mot_des):
        v_in = self.conf_trq_gain * t_mot_des
        return v_in
   
class Mod_Body:
    def __init__(self):
        self.w_wheel = 0
        self.vel_veh = 0                        
        self.the_wheel = 0
        self.t_brake = 0
        self.t_mot_des = 0
        self.t_drag = 0
        self.Body_config()
        self.Dyn_config()
        self.Ts_loc = globals()['Ts']

    def Body_config(self, conf_rw_wheel = 0.3, conf_jw_body = 2, conf_brk_coef = 100, conf_acc_coef = 100, conf_veh_len = 2):
        self.conf_rw_wheel = conf_rw_wheel
        self.conf_jw_body = conf_jw_body
        self.conf_brk_coef = conf_brk_coef
        self.conf_acc_coef = conf_acc_coef
        self.conf_veh_len = conf_veh_len
        
    def Dyn_config(self, conf_airdrag = 4, conf_add_weight = 0):
        self.conf_drag_lon = conf_airdrag
        self.conf_weight_veh = conf_add_weight
    
    def Lon_driven_in(self, u_acc, u_brake, veh_vel = 0):    
        self.t_brake = self.Brake_system(u_brake)
        self.t_mot_des = self.Acc_system(u_acc)      
        self.t_drag = self.Drag_system(veh_vel)
        return self.t_mot_des, self.t_brake, self.t_drag
    
    def Lon_driven_out(self,t_load, t_brake, t_drag):
        self.w_wheel = self.Tire_dynamics(self.w_wheel, t_load, t_brake, t_drag)        
        self.vel_veh = self.w_wheel * self.conf_rw_wheel
        return [self.w_wheel, self.vel_veh]
    
    def Lat_driven(self, u_steer):
        self.the_wheel = self.Lat_dynamics(self.the_wheel, u_steer)
        return self.the_wheel
    
    def Lat_dynamics(self, the_wheel, u_steer):
        the_wheel = the_wheel + self.Ts_loc/0.2*(u_steer - the_wheel)
        return the_wheel

    def Tire_dynamics(self, w_wheel, t_load, t_brake = 0, t_drag = 0):               
        w_wheel = w_wheel + self.Ts_loc/self.conf_jw_body*(t_load - t_drag - t_brake)
        return w_wheel       

    def Brake_system(self, u_brake):        
        if self.w_wheel <= 0:
            t_brake = 0
        else:
            t_brake = u_brake * self.conf_brk_coef       
        return t_brake
    
    def Acc_system(self, u_acc):
        t_mot_des = u_acc * self.conf_acc_coef       
        return t_mot_des
    
    def Drag_system(self, veh_vel):
        t_drag = self.conf_drag_lon * veh_vel
        if t_drag < 0:
            t_drag = 0
        return t_drag

    
class Mod_Veh:    
    def __init__(self,powertrain_model,body_model):        
        self.ModPower = powertrain_model
        self.ModBody = body_model
        self.Ts_loc = globals()['Ts']        
        self.Set_initState()
        
    def Set_initState(self, x_veh = 0, y_veh = 0, s_veh = 0, n_veh = 0, psi_veh = 0):
        self.pos_x_veh = x_veh
        self.pos_y_veh = y_veh
        self.pos_s_veh = s_veh
        self.pos_n_veh = n_veh        
        self.psi_veh = psi_veh   
    
    def Veh_driven(self, u_acc = 0, u_brake = 0, u_steer = 0):
        t_load = self.ModPower.t_load
        w_shaft = self.ModBody.w_wheel
        veh_vel = self.ModBody.vel_veh
        # Lateral motion
        the_wheel = self.ModBody.Lat_driven(u_steer)
        # Longitudinal motion
        # Body_Lon_in --> Powertrain_Motor --> Body_Lon_out
        [t_mot_des, t_brake, t_drag] = self.ModBody.Lon_driven_in(u_acc, u_brake, veh_vel)
        [w_mot, t_mot, t_load] = self.ModPower.Motor_control(t_mot_des, w_shaft)
        [w_wheel, vel_veh] = self.ModBody.Lon_driven_out(t_load, t_brake, t_drag)        
        return vel_veh, the_wheel
        
    def Veh_position_update(self, vel_veh = 0, the_wheel = 0):
        veh_len = self.ModBody.conf_veh_len
        ang_veh = the_wheel + self.psi_veh
        x_dot = vel_veh*cos(ang_veh)
        self.pos_x_veh = self.pos_x_veh + x_dot*self.Ts_loc
        y_dot = vel_veh*sin(ang_veh)
        self.pos_y_veh = self.pos_y_veh + y_dot*self.Ts_loc
        s_dot = vel_veh*cos(the_wheel)
        self.pos_s_veh = self.pos_s_veh + s_dot*self.Ts_loc
        n_dot = vel_veh*sin(the_wheel)
        self.pos_n_veh = self.pos_n_veh + n_dot*self.Ts_loc
        psi_dot = vel_veh/veh_len*the_wheel
        self.psi_veh = self.psi_veh + psi_dot*self.Ts_loc
        return [self.pos_x_veh, self.pos_y_veh, self.pos_s_veh, self.pos_n_veh, self.psi_veh]
#%% 2. Maneuver
    # 1. Driver
    # 2. Behavior(Driver)     # Recognition integrated in behavior
    
class Mod_Driver:    
    def __init__(self):
        self.set_char('Normal')
        
    def set_char(self, DriverChar = 'Normal'):
        if DriverChar == 'Normal':            
            self.set_driver_param(0.5, 0.1, 0,  1, 1, 0, 1.5, 4)
        elif DriverChar == 'Aggressive':
            self.set_driver_param(0.8, 0.15, 0, 1, 1, 0, 1.5, 4)
        elif DriverChar == 'Defensive':
            self.set_driver_param(0.2, 0.05, 0, 1, 1, 0, 1.5, 4)
        else:
            print('Set the driver only = [''Normal'', ''Aggressive'', ''Defensive'']')
            self.set_driver_param(0.5, 0.1, 0, 1, 1, 0, 1.5, 4)
    def set_driver_param(self, P_gain_lon = 0.5, I_gain_lon = 0.1, D_gain_lon = 0, P_gain_lat = 1, I_gain_lat = 1, D_gain_lat = 0, shift_time = 1.5, max_acc = 4):
        self.P_gain_lon = P_gain_lon; self.I_gain_lon = I_gain_lon; self.D_gain_lon = D_gain_lon        
        self.P_gain_lat = P_gain_lat; self.I_gain_lat = I_gain_lat; self.D_gain_lat = D_gain_lat
        self.shift_time = shift_time; self.max_acc = max_acc    
        
class Mod_Behavior:
    def __init__(self, Driver):
        self.stStaticList = type_drvstate()
        self.stDynamicList = type_drvstate()
        self.stStatic = type_drvstate()
        self.stDynamic = type_drvstate()        
        self.Maneuver_config()
        self.Drver_set(Driver)
        self.Ts_Loc = globals()['Ts']
        
    def Lon_control(self,veh_vel_set,veh_vel):                
        # Value initialization
        if not 'timer_count' in locals():
            timer_count = 0
            shift_flag_ac = 'on'
            shift_flag_br = 'on'
        
        trq_set = self.Lon_Controller.Control(veh_vel_set,veh_vel)
        if (trq_set >= 5) & (shift_flag_ac == 'on'):
            stControl = 'acc'
            shift_flag_ac = 'on'
            shift_flag_br = 'off'
            timer_count = 0
        elif (trq_set <= 0) & (shift_flag_br == 'on'):
            stControl = 'brk'
            shift_flag_ac = 'off'
            shift_flag_br = 'on'           
            timer_count = 0
        else:
            stControl = 'idle'
            if (shift_flag_ac == 'on') & (timer_count >= self.Driver.shift_time):
                timer_count = 0
                shift_flag_br = 'on'                
            elif (shift_flag_br == 'on') & (timer_count >= self.Driver.shift_time):
                timer_count = 0
                shift_flag_ac = 'on'
            else:
                timer_count = timer_count + 1                
        # Set value
        if stControl == 'acc':
            acc_out = trq_set/100
            brk_out = 0
        elif stControl == 'brk':
            acc_out = 0
            brk_out = -trq_set/100
        elif stControl == 'idle':
            acc_out = 0
            brk_out = 0
        else:
            acc_out = 0
            brk_out = 0
        self.trq_set_lon = trq_set
        self.u_acc = acc_out
        self.u_brk = brk_out
        return [acc_out, brk_out]
    
    def Lat_control(self,lane_offset, offset_des = 0):
        steer_out = self.Lat_Controller.Control(offset_des,lane_offset)
        self.u_steer = steer_out
        return steer_out
    
    def Drver_set(self, DriverSet):
        self.Driver = DriverSet
        self.Lon_Controller = type_pid_controller(DriverSet.P_gain_lon, DriverSet.I_gain_lon, DriverSet.D_gain_lon)
        self.Lat_Controller = type_pid_controller(DriverSet.P_gain_lat, DriverSet.I_gain_lat, DriverSet.D_gain_lat)                
            
    def Maneuver_config(self, cruise_speed_set = 60, curve_coef = 1500, transition_dis = 20, forecast_dis = 100, cf_dis = 120, lat_off = 0.5):                
        self.conf_cruise_speed_set = cruise_speed_set
        self.conf_curve_coef = curve_coef
        self.conf_transition_dis = transition_dis
        self.conf_forecast_dis = forecast_dis
        self.conf_cf_dis = cf_dis            
        self.conf_lat_off = lat_off
         
    def Static_state_recog(self,static_obj_in, veh_position_s, road_len):
        # Define local state and objectives
        stStatic = type_drvstate()
        forecast_object = type_objective()
        transition_object = type_objective()        
        # Determine map_index (forecasting, transition)
        tmp_cur_index = np.min(np.where(road_len >= veh_position_s))-1        
        tmp_forecast_index = np.min(np.where(road_len >= (tmp_cur_loc + self.conf_forecast_dis)))-1        
        tmp_transition_index = np.min(np.where(S_py >= (tmp_cur_loc + self.conf_transition_dis)))-1       
        # Determine objectives from vehicle location to forecasting range
        for k in range(tmp_cur_index,tmp_forecast_index+1):        
            forecast_object.merg_object(static_obj_in[k].object_class, static_obj_in[k].object_param, static_obj_in[k].object_loc_s)        
        # Determine objectives from transition range to forecasting range                
        for k in range(tmp_transition_index,tmp_forecast_index+1):    
            transition_object.merg_object(static_obj_in[k].object_class, static_obj_in[k].object_param, static_obj_in[k].object_loc_s)                    
        if ('Tl' in forecast_object.object_class):
            tmp_Tl_index = forecast_object.object_class.index('Tl')
            tmp_Tl_param = forecast_object.object_param[tmp_Tl_index]
            tmp_Tl_loc = forecast_object.object_loc_s[tmp_Tl_index]
            if tmp_Tl_param == 'red':
                stStatic.set_state('Tl_stop',tmp_Tl_param,tmp_Tl_loc - tmp_cur_loc)
            else:
                if 'Curve' in transition_object.object_class:
                    tmp_cv_index = np.where(np.array(transition_object.object_class) == 'Curve')[0]
                    tmp_cv_loc = np.mean(np.array(transition_object.object_loc_s)[tmp_cv_index])
                    tmp_cv_param = np.mean(np.array(transition_object.object_param)[tmp_cv_index])
                    stStatic.set_state('Curve',tmp_cv_param,tmp_cv_loc - tmp_cur_loc) 
                else:
                    stStatic.set_state('Cruise')    
        else:
            if 'Curve' in transition_object.object_class:
                tmp_cv_index = np.where(np.array(transition_object.object_class) == 'Curve')[0]
                tmp_cv_loc = np.mean(np.array(transition_object.object_loc_s)[tmp_cv_index])
                tmp_cv_param = np.mean(np.array(transition_object.object_param)[tmp_cv_index])
                stStatic.set_state('Curve',tmp_cv_param,tmp_cv_loc - tmp_cur_loc) 
            else:
                stStatic.set_state('Cruise')
        
        self.stStaticList.add_state(stStatic.state, stStatic.state_param, stStatic.state_reldis)
        return stStatic
    
    def Dynamic_state_recog(self, pre_veh_speed, pre_veh_reldis = 250):
        stDynamic = type_drvstate()        
        if pre_veh_reldis >= self.conf_cf_dis:
            stDynamic.set_state('Cruise')             
        else:
            stDynamic.set_state('Cf', state_reldis_in = pre_veh_reldis)
        self.stDynamicList.add_state(stDynamic.state, stDynamic.state_param, stDynamic.state_reldis)
        return stDynamic
                    
    def Lon_vel_set(self, stStatic, stDynamic):
        # Determination of velocity set from static state
        tmp_state_step_static = stStatic['object_class']        
        if tmp_state_step_static == 'Cruise':
            veh_speed_set_static = self.conf_cruise_speed_set
        elif tmp_state_step_static == 'Tl_stop':
            tmp_state_reldis_step = stStatic['object_rel_dis']
            veh_speed_set_static = self.conf_cruise_speed_set - self.conf_cruise_speed_set*(self.conf_forecast_dis - tmp_state_reldis_step)/self.conf_forecast_dis
        elif tmp_state_step_static == 'Curve':
            tmp_param_step = float(stStatic['object_param'])
            veh_speed_set_static = self.conf_cruise_speed_set - tmp_param_step*self.conf_curve_coef
        # Determination of velocity set from dynamic state
        tmp_state_step_dynamic = stDynamic['object_class']
        if tmp_state_step_dynamic == 'Cruise':
            veh_speed_set_dynamic = self.conf_cruise_speed_set
        else:
            tmp_preveh_vel = stDynamic['object_param']
            veh_speed_set_dynamic = tmp_preveh_vel        
        veh_speed_set = min(veh_speed_set_dynamic,veh_speed_set_static)
        return [veh_speed_set, veh_speed_set_static, veh_speed_set_dynamic]
    
    def Lon_behavior(self,static_obj_in, veh_position_s, road_len, pre_veh_speed = 'None', pre_veh_reldis = 250):
        self.stStatic = self.Static_state_recog(static_obj_in, veh_position_s, road_len)
        self.stDynamic = self.Dynamic_state_recog(pre_veh_speed, pre_veh_reldis)
        veh_speed_set = self.Lon_vel_set(self.stStatic,self.stDynamic)
        [self.acc_out, self.brk_out] = self.Lon_control(self,veh_speed_set,veh_vel)
        return [self.acc_out, self.brk_out]
    
    def Lat_behavior(self, veh_position_x, veh_position_y, road_x, road_y):
        self.stLateral = self.Lateral_state_recog(veh_position_x, veh_position_y, road_x, road_y)
        if self.stLateral.state == 'Right':
            Lane_offset = self.stLateral.state_reldis
        else:
            Lane_offset = -self.stLateral.state_reldis
        self.lane_offset = Lane_offset
        self.steer_out = self.Lat_control(Lane_offset)        
        return self.steer_out    
            
    def Lateral_state_recog(self, veh_position_x, veh_position_y, road_x, road_y):
        stLateral = type_drvstate()
        [lat_offset, direction] = Calc_PrDis(road_x, road_y, [veh_position_x, veh_position_y])
        stLateral.set_state(direction,'None',lat_offset)  
#        if lat_offset <= abs(self.conf_lat_off):
#            stLateral.set_state('None','None',0)
#        else:
#            stLateral.set_state(direction,'None',lat_offset)          
        return stLateral
#%% 3. Environment model
    # Road, static objects
class Mod_Env:    
    def __init__(self, road_array_x_in, road_array_y_in, start_road_len = 0):
        self.road_x = road_array_x_in
        self.road_y = road_array_y_in
        self.object_list = [type_objective() for _ in range(len(road_array_x_in))]
        self.Road_config(start_road_len)
        
    def Road_config(self, start_road_len = 0):
        road_array_x_in = self.road_x
        road_array_y_in = self.road_y
        loc_env_road_s = np.zeros(len(road_array_x_in))
        loc_env_road_s[0] = start_road_len
        loc_env_road_ang = np.zeros(len(road_array_x_in))
        loc_env_road_ang[0] = 0
        for i in range(1,len(road_array_x_in),1):    
            old_pos = [road_array_x_in[i-1],road_array_y_in[i-1]]
            new_pos = [road_array_x_in[i],road_array_y_in[i]]
            loc_env_road_s[i] = loc_env_road_s[i-1] + dist_calc.euclidean(old_pos, new_pos)                        
            loc_env_road_ang[i] = np.arctan((road_array_y_in[i] - road_array_y_in[i-1])/(road_array_x_in[i] - road_array_x_in[i-1]))
        self.road_ang = loc_env_road_ang
        self.road_len = loc_env_road_s        
        self.object_list = self.Road_curve_def(road_array_x_in, road_array_y_in, loc_env_road_s)

    def Obj_add (self, object_in, object_param_in, object_s_location):
        loc_env_road_s = self.road_len
        tmp_s_index = np.min(np.where(loc_env_road_s >= object_s_location)) - 1           
        self.object_list[tmp_s_index].add_object(object_in,object_param_in,object_s_location)        
       
    def Road_curve_def(self, road_array_x_in, road_array_y_in, loc_env_road_s, conf_curve_val = 0.001):
        object_list = [type_objective() for _ in range(len(road_array_x_in))]
        [R_out, x_c_out, y_c_out, circle_index,mr_o,mt_o] = Calc_Radius(road_array_x_in, road_array_y_in, 3)        
        tmp_Curve = 1/R_out
        tmp_Curve_Filt = Filt_MovAvg(tmp_Curve,3)
        tmp_Curve_index = np.arange(len(road_array_x_in))[tmp_Curve_Filt >= conf_curve_val]
        self.road_curve = tmp_Curve
        for i in range(len(tmp_Curve_index)):
            tmp_s_index = tmp_Curve_index[i]            
            object_list[tmp_s_index].add_object('Curve',tmp_Curve[tmp_s_index],loc_env_road_s[tmp_s_index])            
        return object_list    
#%% 4. Simulation utilities
class type_DataLog:
    def __init__(self, NameSpaceList = []):
        if len(NameSpaceList) == 0:
            print('Should set name space : SetNameSpace(List)')
            self.NameSet = NameSpaceList
            self.DataProfile = {}
        else:
            self.NameSet = NameSpaceList
            self.DataProfile = {}
            for i in NameSpaceList:                
                self.DataProfile.update({i:[]})                 
                
    def SetNameSpace(self, NameSpaceList):        
        self.NameSet = NameSpaceList
        for i in NameSpaceList:
            self.DataProfile.update({i:[]})    
            
    def StoreData(self, DataSet):
        if len(self.NameSet) != len(DataSet):
            print('Data length should same to Name set length')
        else:
            for i in range(len(self.NameSet)):
                self.DataProfile[self.NameSet[i]].append(DataSet[i])
    def get_profile_value(self, get_name_set = []):
        if len(get_name_set) == 0:
            get_name_set = self.NameSet        
        return_profile_list = []
        for i in get_name_set:
            return_profile_list.append(self.DataProfile[i])
        return return_profile_list
    def get_profile_value_one(self, get_name_set):            
        return self.DataProfile[get_name_set]
#%%  ----- test ground -----
if __name__ == "__main__":
    #%% power_train + long_dyna test
#    kona_power = Mod_PowerTrain()
#    kona_body = Mod_Body()
#    
#    v_in = 36
#    w_wheel = 0
#    Ts = 0.01
#    sim_time = 20
#    
#    result = {'w_motor':[],'w_wheel':[],'vel_veh':[],'t_motor':[],'t_load':[]}
#    for sim_step in range(sim_time):
#        [w_motor, t_motor, t_load] = kona_power.Motor_driven(v_in, w_wheel)
#        [w_wheel, vel_veh] = kona_body.Lon_driven(t_load)
    #%% Environment and static state recognition test
    # load midan data
    RoadData = io.loadmat('road_data_midan.mat')
    X_in = np.reshape(RoadData['sn_X'],383)
    Y_in = np.reshape(RoadData['sn_Y'],383)
    # define model with data
    env_midan = Mod_Env(X_in,Y_in)
    # add static obstacle - traffic light
    env_midan.Obj_add('Tl','red',2200)
    #env_midan.Obj_add('Tl','red',300)
    
    # Configuration variables
    tmp_forecast_dis = 100
    tmp_transition_dis = 10
    tmp_cruise_vel = 100
    
    S_py = env_midan.road_len
    veh_loc_s = np.arange(15,max(S_py)-tmp_forecast_dis,10)
    obj = env_midan.object_list
    
    tmp_object_list = [obj[i].object_class for i in range(len(obj))]
    #%%
    vel_set = np.zeros(len(veh_loc_s))
    
    stStaticList = type_drvstate()
    stStatic = type_drvstate()
    # Static_Recognition (Veh_pos_s, obj_list) and velocity profiles
    for i in range(len(veh_loc_s)):
        tmp_cur_loc = veh_loc_s[i]
        tmp_cur_index = np.min(np.where(S_py >= tmp_cur_loc))-1
        tmp_cur_loc_s = S_py[tmp_cur_index]
        tmp_forecast_index = np.min(np.where(S_py >= (tmp_cur_loc + tmp_forecast_dis)))-1
        tmp_forecast_loc = tmp_cur_loc + tmp_forecast_dis
        tmp_forecast_loc_s = S_py[tmp_forecast_index]
        tmp_transition_index = np.min(np.where(S_py >= (tmp_cur_loc + tmp_transition_dis)))-1
        
        forecast_object = type_objective()
        transition_object = type_objective()    
        for k in range(tmp_cur_index,tmp_forecast_index+1):        
            forecast_object.merg_object(obj[k].object_class, obj[k].object_param, obj[k].object_loc_s)        
                
        for k in range(tmp_transition_index,tmp_forecast_index+1):    
            transition_object.merg_object(obj[k].object_class, obj[k].object_param, obj[k].object_loc_s)        
            
        if ('Tl' in forecast_object.object_class):
            tmp_Tl_index = forecast_object.object_class.index('Tl')
            tmp_Tl_param = forecast_object.object_param[tmp_Tl_index]
            tmp_Tl_loc = forecast_object.object_loc_s[tmp_Tl_index]
            if tmp_Tl_param == 'red':
                stStatic.set_state('Tl_stop',tmp_Tl_param,tmp_Tl_loc - tmp_cur_loc)
            else:
                if 'Curve' in transition_object.object_class:
                    tmp_cv_index = np.where(np.array(transition_object.object_class) == 'Curve')[0]
                    tmp_cv_loc = np.mean(np.array(transition_object.object_loc_s)[tmp_cv_index])
                    tmp_cv_param = np.mean(np.array(transition_object.object_param)[tmp_cv_index])
                    stStatic.set_state('Curve',tmp_cv_param,tmp_cv_loc - tmp_cur_loc) 
                else:
                    stStatic.set_state('Cruise')
    
        else:
            if 'Curve' in transition_object.object_class:
                tmp_cv_index = np.where(np.array(transition_object.object_class) == 'Curve')[0]
                tmp_cv_loc = np.mean(np.array(transition_object.object_loc_s)[tmp_cv_index])
                tmp_cv_param = np.mean(np.array(transition_object.object_param)[tmp_cv_index])
                stStatic.set_state('Curve',tmp_cv_param,tmp_cv_loc - tmp_cur_loc) 
            else:
                stStatic.set_state('Cruise')
        
        stStaticList.add_state(stStatic.state, stStatic.state_param, stStatic.state_reldis)
    #%% 
    
    #%% Plot environment
    road_x = env_midan.road_x
    road_y = env_midan.road_y
    road_curve = env_midan.road_curve
    road_len = env_midan.road_len
    
    veh_vel = np.reshape(RoadData['sn_Ego_Velocity'],383)
    
    tmp_state = np.array(stStaticList.state)
    tmp_state_param = np.array(stStaticList.state_param)
    tmp_state_reldis = np.array(stStaticList.state_reldis)
    
    index_curve = np.where(tmp_state=='Curve')
    tmp_state_param_cv = np.float64(tmp_state_param[index_curve])
    
    fig = plt.figure(figsize = (12,6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(3,2,2)
    ax3 = plt.subplot(3,2,4)
    ax4 = plt.subplot(3,2,6)
    ax1.plot(road_x,road_y)
    ax1.plot(road_x[index_curve],road_y[index_curve],'.')
    ax2.plot(road_len,veh_vel)
    ax2.plot(road_len[index_curve],veh_vel[index_curve],'.')
    ax3.plot(road_len,road_curve)
    ax3.plot(road_len[index_curve],road_curve[index_curve],'.')
    ax4.plot(road_len,road_curve)
    ax4.plot(road_len[index_curve],tmp_state_param_cv,'.')
    
    #%% Vehicle lon speed set using 
    conf_cruise_set = 60
    conf_curve_coef = 1500
    veh_speed_set = np.zeros(len(veh_loc_s))
    for i in range(len(tmp_state)):
        tmp_state_step = tmp_state[i]    
        if tmp_state_step == 'Cruise':
            veh_speed_set[i] = conf_cruise_set
        elif tmp_state_step == 'Tl_stop': 
            tmp_state_reldis_step = float(tmp_state_reldis[i])
            veh_speed_set[i] = conf_cruise_set - conf_cruise_set*(tmp_forecast_dis-tmp_state_reldis_step)/tmp_forecast_dis
        elif tmp_state_step == 'Curve':
            tmp_param_step = float(tmp_state_param[i])
            veh_speed_set[i] = conf_cruise_set - tmp_param_step*conf_curve_coef           
        
    #%%
    S_Py = np.zeros(len(X_in))
    S_Py[0] = 10
    for i in range(1,len(X_in),1):    
        old_pos = [X_in[i-1],Y_in[i-1]]
        new_pos = [X_in[i],Y_in[i]]
        S_Py[i] = S_Py[i-1] + dist_calc.euclidean(old_pos, new_pos)
#%%
    road_x = np.arange(1,2000,10)        
    road_y = 100*np.ones(len(road_x))
    plt.plot(road_x, road_y)
    
    veh_x = np.arange(3,2000,10)        
    veh_y = 100*np.ones(len(veh_x))
    y_sin = 10*np.sin(np.arange(0,20,0.2))
    veh_y[100:] = 95+y_sin
            
    en_test = Mod_Env(road_x,road_y)
    dis_list = []
    dir_list = []
    veh_an = []
    road_an = []
    #%%
RoadData = io.loadmat('road_data_amsa.mat')
 # Powertrain import and configuration
kona_power = Mod_PowerTrain()
# ~~~~~
# Bodymodel import and configuration
kona_body = Mod_Body()
kona_body.conf_veh_len = 1
# ~~~~
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_body)
# ~~~~
# Driver model
drv_kyunghan = Mod_Driver()
# ~~~~
# Behavior model
beh_steer = Mod_Behavior(drv_kyunghan)
# Road model
#road_x = np.arange(1,4000,1)        
#road_sin = 100 + 30*np.sin(np.arange(0, len(road_x)*0.8*0.003, 0.003))
#road_y = np.concatenate((100*np.ones(int(len(road_x)*0.2)), road_sin))
road_x = RoadData['sn_X']
road_y = RoadData['sn_Y']
# 4.2 Simulation config
Ts = 0.01
sim_time = 20
sim_time_range = np.arange(0,sim_time,0.01)
veh_vel = 0 # Initial vehicle speed
kona_vehicle.Set_initState(x_veh = road_x[2], y_veh = road_y[2], psi_veh = atan((road_y[2] - road_y[1]))/(road_x[2] - road_x[1]))
u_speed_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 10 * np.ones(int(len(sim_time_range)*0.9))))

# ----------------------------- select input set ---------------------------
Input_index = 3
if Input_index == 0:
    # Driver = normal
    drv_kyunghan.set_char('Normal')
    beh_steer.Drver_set(drv_kyunghan)
elif Input_index == 1:
    # Driver = aggressive
    drv_kyunghan.set_char('Aggressive')
    beh_steer.Drver_set(drv_kyunghan)
elif Input_index == 2:
    # Driver = defensive
    drv_kyunghan.set_char('Defensive')
    beh_steer.Drver_set(drv_kyunghan)
elif Input_index == 3:
    # Driver = new driver with config parameter
    drv_kyuhwan = Mod_Driver()
    drv_kyuhwan.P_gain_lon = 0.2
    drv_kyuhwan.I_gain_lon = 0.05
    drv_kyuhwan.I_gain_lat = 0
    drv_kyuhwan.P_gain_lat = 0
    beh_steer.Drver_set(drv_kyuhwan)
else:
    print('입력을 똑바로 하세요 ~_~')
    
# 4.3 Simulation
# Set logging data
sim4 = type_DataLog(['Veh_Vel','Acc_in','Brk_in','Trq_set','Steer_in','Wheel_theta','LaneOff'])
sim4_veh = type_DataLog(['PosX','PosY'])

for sim_step in range(len(sim_time_range)):        
    # Arrange vehicle position
    pos_x = kona_vehicle.pos_x_veh
    pos_y = kona_vehicle.pos_y_veh
    # Arrange behavior input
    u_speed_set = u_speed_val[sim_step]    
    # Behavior control    
    [u_acc_in, u_brk_in] = beh_steer.Lon_control(u_speed_set, veh_vel)
    u_steer_in = beh_steer.Lat_behavior(pos_x,pos_y,road_x,road_y)     
    # Vehicle model sim
    [veh_vel, the_wheel] = kona_vehicle.Veh_driven(u_acc = u_acc_in, u_brake = u_brk_in, u_steer = u_steer_in)
    [pos_x, pos_y, pos_s, pos_n, psi_veh] = kona_vehicle.Veh_position_update(veh_vel, the_wheel)       
    # Store data
    sim4.StoreData([veh_vel, u_acc_in, u_brk_in, kona_vehicle.psi_veh, u_steer_in, the_wheel, beh_steer.lane_offset])
    sim4_veh.StoreData([pos_x,pos_y])

[sim4_veh_vel, sim4_u_acc, sim4_u_brk, sim4_trq_set, sim4_steer, sim4_wheel, sim4_laneoff] = sim4.get_profile_value(['Veh_Vel','Acc_in','Brk_in','Trq_set','Steer_in','Wheel_theta','LaneOff'])
[sim4_veh_x, sim4_veh_y] = sim4_veh.get_profile_value(['PosX','PosY'])

# 4.4 Reulst plot
fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)
ax1.plot(road_x, road_y)
ax1.plot(road_x[6442], road_y[6442],'o')
ax1.plot(road_x[0], road_y[0],'o')
ax1.plot(sim4_veh_x, sim4_veh_y);
ax1.set_xlabel('X position [m]');ax1.set_ylabel('Y position [m]');ax1.axis('equal')
ax2.plot(sim_time_range, sim4_trq_set)
ax3.plot(sim_time_range, sim4_wheel,label = 'Wheel')
#ax3.plot(sim_time_range, sim4_steer,label = 'Str')


#%%

Data_Array_x = road_x
Data_Array_y = road_y
Current_point = [pos_x, pos_y]

dis_array = np.sqrt(np.square(Data_Array_x - Current_point[0]) + np.square(Data_Array_y - Current_point[1]))    
min_index = np.argmin(dis_array)    
tmp_c = dis_array[min_index]

if dis_array[min_index-1] <= dis_array[min_index+1]:
    tmp_b = dis_array[min_index-1]
    tmp_a = dist_calc.euclidean([Data_Array_x[min_index], Data_Array_y[min_index]], [Data_Array_x[min_index-1], Data_Array_y[min_index-1]])
    road_an = atan((Data_Array_y[min_index] - Data_Array_y[min_index-1])/(Data_Array_x[min_index] - Data_Array_x[min_index-1]))
    if Data_Array_x[min_index] - Data_Array_x[min_index-1] < 0:
        road_an = road_an + pi        
    veh_an = atan((Current_point[1]-Data_Array_y[min_index-1])/(Current_point[0]-Data_Array_x[min_index-1]))
    if Current_point[0] - Data_Array_x[min_index-1] < 0:
        veh_an = veh_an + pi        
else:
    tmp_b = dis_array[min_index+1]
    tmp_a = dist_calc.euclidean([Data_Array_x[min_index], Data_Array_y[min_index]], [Data_Array_x[min_index+1], Data_Array_y[min_index+1]])
    road_an = atan((Data_Array_y[min_index+1] - Data_Array_y[min_index])/(Data_Array_x[min_index+1] - Data_Array_x[min_index]))
    veh_an = atan((Current_point[1]-Data_Array_y[min_index])/(Current_point[0]-Data_Array_x[min_index]))

if (tmp_a**2 + tmp_c**2 - tmp_b**2)/(2*tmp_a*tmp_c) >= 1:
    tmp_dem = 1
elif (tmp_a**2 + tmp_c**2 - tmp_b**2)/(2*tmp_a*tmp_c) <= -1:
    tmp_dem = -1
else:
    tmp_dem = (tmp_a**2 + tmp_c**2 - tmp_b**2)/(2*tmp_a*tmp_c)
    
if tmp_c == 0:
    tmp_An = 0
else:
    tmp_An = acos(tmp_dem)
    
tmp_n = sin(tmp_An)*tmp_c
if veh_an >= road_an:
    tmp_dir = 'Right'
else:
    tmp_dir = 'Left'            
    
print([tmp_dir, veh_an, road_an])
#return tmp_n, tmp_dir
