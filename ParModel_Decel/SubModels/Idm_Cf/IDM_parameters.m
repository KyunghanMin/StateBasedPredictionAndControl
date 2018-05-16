clc; close all; clear all;
%% Define the value of the parameters in a car-following model
a_max = 3;                  % The maximum vehicle acceleration
v_0 = 30;                   % The velocity the vehicle would drive at in free traffic
T = 1.5;                     % The minimum vehicle acceleration
b = 1;                      % The comfortable braking deceleration
s_0 = 25;                    % The minimum desired net distance. A car can't move if the distance from the car in the front is not at least s0
delta = 4;                  % The acceleration exponent
length = 5;               % The length of the ego vehicle
