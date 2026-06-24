%
%% urdfviewer.m
%
% The purpose of this code is to provide a visualization step before loading the model
% in Gazebo. This is just to ensure that the urdf models generated in
% SW2URDF plugin are generated properly.
%

% Open Manipulator X6 arm
% cd('C:\Users\Lanie\Documents\Ri-One\Projects\Open Manipulator X6\complete_arm_assembly\urdf')
% model = importrobot('complete_arm_assembly.urdf');


% Paperbag Model
cd('C:\Users\Lanie\Documents\Ri-One\Projects\Open Manipulator X6\paperbag\urdf')
model = importrobot('paperbag.urdf');
figure
show(model,'Frames','on');
axis equal
