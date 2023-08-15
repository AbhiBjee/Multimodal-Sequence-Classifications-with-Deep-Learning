% Created by Abhinaba Bhattacharjee, Purdue University, 6/29/2023

% This code is used to visualise the Raw values of sensor data in the form
% of graphs and can be validated with the motion sensing animator and the
% force reaction vector animator to observe manually segmented data chunks
% to verify  triaxial force and hand motion sequences.

clear all;
clc;
 %% Select the file for analysis 
 %Copy and paste the "Clinical Data" folder to MATLAB's current Working Directory 
 % Select the .csv file in the "Clinical Data" folder to observe the hand
 % manipulation mootions
 
 [file,path] = uigetfile('/Users/user/*.csv');
 csvPath = fullfile(path, file);
 Data = readtable(csvPath);

%% Load and Store Data

time = table2array(Data(:,1));

FrcX = table2array(Data(:,2));
FrcY = table2array(Data(:,3));
FrcZ = table2array(Data(:,4));
FrcRez = table2array(Data(:,5));

YawGeo = table2array(Data(:,6));
PitchGeo = table2array(Data(:,7));
RollGeo = table2array(Data(:,8));

AccX = table2array(Data(:,9));
AccY = table2array(Data(:,10));
AccZ = table2array(Data(:,11));
GyroRez= table2array(Data(:,14));

%% Filtering Yaw Pitch Roll Data using moving average filter
windowSize = 25; a=1;
b = (1/windowSize)*ones(1,windowSize);
YawGeoFilt = filter(b,a,YawGeo);
PitchGeoFilt = filter(b,a,PitchGeo);
RollGeoFilt = filter(b,a,RollGeo);

%% Plot Data in Subplots


% Use the commented section below to plot the Force,
% Accelerometer and the gyro data for graphical observation

plt1 = subplot(3,1,1)
plot(time,FrcX,'r', time,FrcY, 'g', time,FrcZ, 'b', time,FrcRez,'k');
title ('Force Chart (XYZ)');
ylabel('Newtons');
xlabel('Time (sec)');
grid on;

plt2 = subplot(3,1,2)
plot(time,AccX,'r', time,AccY, 'g', time,AccZ, 'b');
title ('Acceleration Chart (XYZ)');
ylabel('G-(9.8m/s^2)');
xlabel('Time (sec)');
grid on;


plt3 = subplot(3,1,3)
plot(time,GyroRez,'k');
title ('Gyro RMS');
ylabel('Degrees/sec');
xlabel('Time (sec)');
grid on;

linkaxes([plt1,plt2,plt3],'x');

% Use the commented section below to plot the Yaw, Pitch and Roll data 
% for graphical observation.

% % % % plot(time,YawGeoFilt,'r', time,PitchGeoFilt, 'g', time,RollGeoFilt, 'b');
% % % % title ('Angular Orientations (YPR - RGB)');
% % % % ylabel('Degrees');

% Use the commented section below to plot the only the triaxial Force data 
% for graphical observation.

% % % % plot(time,FrcX,'r', time,FrcY, 'g', time,FrcZ, 'b', time,FrcRez,'k');
% % % % title ('Force Chart (XYZ)');
% % % % ylabel('Newtons');
% % % % xlabel('Time (sec)');
% % % % grid on;

%% Manually Define the Start and stop timestamps and find their respective ids 
% Please select the timestamps from the x axis of the graph data as
% observed in the plots or sub plots in the last section

StrtVal = 51.72;
StopVal = 63.193;
startID = find(time == StrtVal);
stopID = find(time == StopVal);


%% Animate Data using orientation viewer
% There are two for loop intializations in this section for 3D animation of
% the motion sequence data using unit quaternions.
% The commented for loop initialization may be used for animating the 
% manually chosen data chunk, whereas the for loop initialization can also
% be used for visualization of the whole data sequence from start to end 
% of file.

viewer = HelperOrientationViewer('Title',{'QSTM AHRS Display'});

% use start and stop ids to animate specific portions/chunk of the data

for i=startID:stopID

% use the total time window for animating the whole datafile.

%for i=1:size(time)

    eul = deg2rad([(YawGeoFilt(i)) (PitchGeoFilt(i)) (RollGeoFilt(i))]);
    %quat = quaternion([(YawGeo(i)) (PitchGeo(i)) (RollGeo(i))],'eulerd','ZYX','frame');
    pause(0.01);
    qZYX = eul2quat(eul,'ZYX');
    rotators = quaternion(qZYX);
    %rotators2 = quaternion(quat);
    viewer(rotators)

end

%% Animate Data using Force Direction Cosines or force vector

% Use Custom arrow function - arrow3d(x,y,z,head_frac,radii,radii2,colr)
%
%       where x=[x_start, x_end]; y=[y_start, y_end];z=[z_start,z_end];
%       head_frac = fraction of the arrow length where the head should  start
%       radii = radius of the arrow
%       radii2 = radius of the arrow head (defult = radii*2)
%       colr =   color of the arrow, can be string of the color name, or RGB vector  (default='blue')

figure(1) 

% There are two for loop intializations for 3D reaction force vector 
% visulization this section. The commented for% loop initialization may be
% used for animating the manually chosen data chunk, whereas the for loop 
% initialization can also be used for visualization of the whole data 
% sequence from start to end of file.

% use start and stop ids to animate specific portions of the data

for i=startID:stopID

% use the total time window for animating the whole datafile.

%for i=1:size(time)

    x = [0 FrcX(i)];
    y = [0 FrcY(i)];
    z = [0 FrcZ(i)];
    linRad = FrcRez(i)./20;
    arrowHeadRad = FrcRez(i)./10;    

    %pause(0.01);
    
    %plt.LineWidth = FrcRez(i)./20;
    arrow3d(x,y,z,.8,linRad,arrowHeadRad,'k'); 

    xlim ([-20 20]);
    ylim ([-20 20]);
    zlim ([-2 20]);   

    xlabel('X-Axis (Newton)');
    ylabel('Y-Axis (Newton)');
    zlabel('Z-Axis (Newton)');
    grid on;

    title ('Reaction Force Vector viewer');

    %drawnow update;
    drawnow ();

end

