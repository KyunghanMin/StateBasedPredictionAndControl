%% 4. Design Curvature based Speed Model (Composed with 2 sections) (Validation)
%%      [Process 1] Set range of sections
CurveInit = round(3700/Sampling);
CurveEnd = round(3770/Sampling);
% RangeEye2Road = 24;
SecondCurve_Total = CurveInit:CurveEnd;
SC_Tot_Curvature = double(flt_curvature(SecondCurve_Total));
SC_Tot_Speed =double(sn_Ego_Velocity(SecondCurve_Total));
SC_Tot_V0 = SC_Tot_Speed(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[SC_Curve_max,SC_Curve_maxIdx] = max(SC_Tot_Curvature);
Section_range = SC_Curve_maxIdx;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SecondCurve_Entry = CurveInit:CurveInit+Section_range-1;
SC_Ent_Curvature = double(flt_curvature(SecondCurve_Entry));
SC_Ent_Speed =double(sn_Ego_Velocity(SecondCurve_Entry));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SecondCurve_Exit = CurveInit+Section_range:CurveEnd;
SC_Exit_Curvature = double(flt_curvature(SecondCurve_Exit));
SC_Exit_Speed =double(sn_Ego_Velocity(SecondCurve_Exit));
% %%      [Process 2] Design Curv based Speed Mdl
% SC_Ent_V0 = SC_Ent_Speed(1);
% fun1 = @(k,SC_Curvature)SC_Ent_V0-k(1)*exp(k(2)*SC_Ent_V0*SC_Curvature);
% k0 = [0,0];
% k_ent = lsqcurvefit(fun1,k0,SC_Ent_Curvature,SC_Ent_Speed);
% CurMdlSpd_Ent = fun1(k_ent,SC_Ent_Curvature);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SC_Exit_V0 = CurMdlSpd_Ent(end);
% fun3 = @(k,SC_Curvature)SC_Exit_V0-k(1)*exp(k(2)*SC_Exit_V0*SC_Curvature);
% k_exit = lsqcurvefit(fun3,k0,SC_Exit_Curvature,SC_Exit_Speed);
% CurMdlSpd_Exit = fun3(k_exit,SC_Exit_Curvature);
%%      [Process 3] Result Plot
figure(900)
% set(gcf, 'color','w','Units', 'Normalized', 'OuterPosition', [0 0.05 1 0.95]);
set(gcf, 'color','w','pos',[10 10 1300 700]);
subplot(2,2,1)
pltIndxRoad = Sampling*SecondCurve_Entry;
plot(pltIndxRoad,SC_Ent_Speed,'black','linewidth',3);
hold on;
plot(pltIndxRoad,fun1(k_ent,SC_Ent_Curvature),'magenta','linewidth',3);
hold off;
ylabel('Vehicle Speed [km/h]');
legend('true','Curvature Model');
xlabel('Roadway position [m]');
title('Curve Speed model (Enter section)');
grid on;
subplot(2,2,2)
pltIndxRoad = Sampling*SecondCurve_Exit;
plot(pltIndxRoad,SC_Exit_Speed,'black','linewidth',3);
hold on;
plot(pltIndxRoad,fun3(k_exit,SC_Exit_Curvature),'magenta','linewidth',3);
hold off;
ylabel('Vehicle Speed [km/h]');
legend('true','Curvature Model');
xlabel('Roadway position [m]');
title('Curve Speed model (Exit section)');
grid on;

CurvMdlSpd_2parts = [CurMdlSpd_Ent CurMdlSpd_Exit];
subplot(2,2,3:4)
pltIndxRoad = Sampling*SecondCurve_Total;
plot(pltIndxRoad,SC_Tot_Speed,'black','linewidth',3);
hold on;
plot(pltIndxRoad,CurvMdlSpd_2parts,'magenta','linewidth',3);
hold off;
xlabel('Roadway Position [m]');
ylabel('Vehicle Speed [km/h]');
title('Curvature based Speed Model');
legend('true','Curvature Model');
grid on;
% Calculation R2 RMSE
[R2_Mdl_2sect, RMSE_Mdl_2sect] = Cal_RMSE_RSQUARE(SC_Tot_Speed,CurvMdlSpd_2parts);
