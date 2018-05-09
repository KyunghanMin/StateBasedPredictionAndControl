clear all; clc;
%% 폴더경로
cd('C:\CM_Projects\Deceleartion_prediction\src_cm4sl');
run('cmenv.m')
%% 예측 속도 profile / Sampling Time
open('generic')
CM_Simulink;
%% Open the TestRun file
cmguicmd('LoadTestRun "Simulation_1"')

%% Define the speed set of the preceding vehicle
% Speed_high = [50, 60, 70, 80, 90, 100];
% Speed_low = [20, 30, 40; 30, 40, 50; 40, 50, 60; 50, 60, 70; 60, 70, 80; 70, 80, 90];
Speed_variation = xlsread('Speed_variation.xlsx');

%% CarMaker Simulation
Data = [];

for Variation_idx = 1 : size(Speed_variation,1)
        Speed_high_change0 = sprintf('IFileModify TestRun "Traffic.0.Man.0.LongDyn" "auto %d"',Speed_variation(Variation_idx,1));
        %         Speed_high_change2 = sprintf('IFileModify TestRun "Traffic.0.Man.2.LongDyn" "auto %d"',Speed_high(High_idx));
        %         Speed_high_change4 = sprintf('IFileModify TestRun "Traffic.0.Man.4.LongDyn" "auto %d"',Speed_high(High_idx));
        %         Speed_high_change6 = sprintf('IFileModify TestRun "Traffic.0.Man.6.LongDyn" "auto %d"',Speed_high(High_idx));
        %         Speed_high_change8 = sprintf('IFileModify TestRun "Traffic.0.Man.8.LongDyn" "auto %d"',Speed_high(High_idx));
        %         Speed_high_change10 = sprintf('IFileModify TestRun "Traffic.0.Man.10.LongDyn" "auto %d"',Speed_high(High_idx));
        %         Speed_high_change12 = sprintf('IFileModify TestRun "Traffic.0.Man.12.LongDyn" "auto %d"',Speed_high(High_idx));
        
        Speed_low_change1 = sprintf('IFileModify TestRun "Traffic.0.Man.1.LongDyn" "auto %d"',Speed_variation(Variation_idx,2));
        %         Speed_low_change3 = sprintf('IFileModify TestRun "Traffic.0.Man.3.LongDyn" "auto %d"',Speed_low(High_idx,Low_idx));
        %         Speed_low_change5 = sprintf('IFileModify TestRun "Traffic.0.Man.5.LongDyn" "auto %d"',Speed_low(High_idx,Low_idx));
        %         Speed_low_change7 = sprintf('IFileModify TestRun "Traffic.0.Man.7.LongDyn" "auto %d"',Speed_low(High_idx,Low_idx));
        %         Speed_low_change9 = sprintf('IFileModify TestRun "Traffic.0.Man.9.LongDyn" "auto %d"',Speed_low(High_idx,Low_idx));
        %         Speed_low_change11 = sprintf('IFileModify TestRun "Traffic.0.Man.11.LongDyn" "auto %d"',Speed_low(High_idx,Low_idx));
        
        cmguicmd(Speed_high_change0);
        %         cmguicmd(Speed_high_change2);
        %         cmguicmd(Speed_high_change4);
        %         cmguicmd(Speed_high_change6);
        %         cmguicmd(Speed_high_change8);
        %         cmguicmd(Speed_high_change10);
        %         cmguicmd(Speed_high_change12);
        
        cmguicmd(Speed_low_change1);
        %         cmguicmd(Speed_low_change3);
        %         cmguicmd(Speed_low_change5);
        %         cmguicmd(Speed_low_change7);
        %         cmguicmd(Speed_low_change9);
        %         cmguicmd(Speed_low_change11);
        %
        cmguicmd('IFileFlush');
        
        sim('generic'); pause(1); disp(CMData.LastResultFName);
        Data_temp = cmread(CMData.LastResultFName);
        Data = [Data; Data_temp];
end

%% Save the simulation data
cd('C:\Users\Gyubin\OneDrive\문서\ACE Lab\GRG\2018 CX\Deceleration_prediction_car_following\CarMaker_simulation')
save Simulation_data.mat Data

