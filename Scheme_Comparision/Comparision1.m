%% Scheme 1: Hybrid Orientation and Position Control
% Comparison implementation based on:
% Z. Xie and L. Jin, "Hybrid control of orientation and position for
% redundant manipulators using neural network."
%
% This script evaluates pose tracking of a PUMA 560 manipulator using the
% dynamic neural network (DNN) formulation reported in the reference above.
% The desired task consists of a three-dimensional astroid position
% trajectory and a time-varying ZYX Euler-angle orientation trajectory.
%
% Requirements:
%   1. MATLAB.
%   2. Peter Corke's Robotics Toolbox for MATLAB (mdl_puma560, fkine,
%      jacob0, and ikcon).
%   3. eul2rotm with ZYX Euler-angle convention.
%
% Reproducibility note:
%   All controller, trajectory, integration, and constraint parameters used
%   in this comparison are defined explicitly in Sections 1-5 below.

clear; clc; close all;

mdl_puma560;
robot = p560;
num_joints = robot.n;

sim_time = 20;             % Total simulation time [s]
dt = 0.005;                % Outer-loop sampling interval [s]
time = 0:dt:sim_time;
num_samples = numel(time);

trajectory_scale = 0.1;
position_center = [0.6891; -0.0069; 0.1778];

trajectory_omega = 2*pi/sim_time;
trajectory_phase = trajectory_omega * time;
cos_phase = cos(trajectory_phase);
sin_phase = sin(trajectory_phase);

desired_position = [ ...
    position_center(1) + trajectory_scale*cos_phase.^3 - trajectory_scale; ...
    position_center(2) + trajectory_scale*sin_phase.^3; ...
    position_center(3) + trajectory_scale*sin_phase.^3];

desired_linear_velocity = [ ...
    -3*trajectory_scale*trajectory_omega .* cos_phase.^2 .* sin_phase; ...
     3*trajectory_scale*trajectory_omega .* sin_phase.^2 .* cos_phase; ...
     3*trajectory_scale*trajectory_omega .* sin_phase.^2 .* cos_phase];

yaw_offset = 1.12;
pitch_offset = 0.65;
roll_offset = 0.10;

desired_yaw = 0.05*cos(time) + yaw_offset;
desired_pitch = -0.032*sin(time) + pitch_offset;
desired_roll = 0.015*sin(time) + roll_offset;

desired_yaw_rate = -0.05*sin(time);
desired_pitch_rate = -0.032*cos(time);
desired_roll_rate = 0.015*cos(time);

desired_rotation = cell(num_samples, 1);
desired_angular_velocity = zeros(3, num_samples);

for k = 1:num_samples
    desired_rotation{k} = eul2rotm( ...
        [desired_yaw(k), desired_pitch(k), desired_roll(k)], 'ZYX');

    % Map ZYX Euler-angle rates to body angular velocity, then express the
    % angular velocity in the base frame.
    roll_k = desired_roll(k);
    pitch_k = desired_pitch(k);

    euler_rate_to_body_omega = [ ...
        1, 0, -sin(pitch_k); ...
        0, cos(roll_k),  sin(roll_k)*cos(pitch_k); ...
        0, -sin(roll_k), cos(roll_k)*cos(pitch_k)];

    body_angular_velocity = euler_rate_to_body_omega * [ ...
        desired_roll_rate(k); ...
        desired_pitch_rate(k); ...
        desired_yaw_rate(k)];

    desired_angular_velocity(:, k) = ...
        desired_rotation{k} * body_angular_velocity;
end

initial_position_error = [0.0182; -0.0071; -0.0184];

initial_pose = eye(4);
initial_pose(1:3, 1:3) = desired_rotation{1};
initial_pose(1:3, 4) = desired_position(:, 1) + initial_position_error;

initial_joint_seed = [0; -pi/4; pi/2; 0; pi/4; 0];
joint_position = robot.ikcon(initial_pose, initial_joint_seed')';

position_gain = 15;
orientation_gain = 1;

dnn_mu = 25;              
dnn_nu = 15;               

num_inner_steps = 40;
inner_step = dt / num_inner_steps;

% Joint constraints.
joint_velocity_limit = deg2rad(90);  % [rad/s]
joint_position_min = robot.qlim(:, 1);
joint_position_max = robot.qlim(:, 2);

% Optional joint-margin-dependent tightening of the velocity bounds.
use_dynamic_joint_margin = false;
joint_margin_gain = 4.0;

vartheta = zeros(num_joints, 1);
y_aux = zeros(6, 1);

%% Preallocate data logs
position_error_log = zeros(3, num_samples);
orientation_error_log = zeros(3, num_samples);

actual_position_log = zeros(3, num_samples);
desired_position_log = desired_position;

desired_euler_log = [desired_yaw; desired_pitch; desired_roll];
actual_euler_log = zeros(3, num_samples);

joint_position_log = zeros(num_joints, num_samples);
joint_velocity_log = zeros(num_joints, num_samples);

task_velocity_command_log = zeros(6, num_samples);
task_velocity_actual_log = zeros(6, num_samples);

%% Main simulation loop
for k = 1:num_samples
    % Current end-effector pose.
    current_pose = robot.fkine(joint_position).T;
    current_rotation = current_pose(1:3, 1:3);
    current_position = current_pose(1:3, 4);

    % Geometric Jacobian expressed in the base frame.
    geometric_jacobian = robot.jacob0(joint_position);
    task_jacobian = [ ...
        geometric_jacobian(1:3, :); ...
        geometric_jacobian(4:6, :)];

    % Desired end-effector pose and velocity.
    desired_position_k = desired_position(:, k);
    desired_linear_velocity_k = desired_linear_velocity(:, k);
    desired_rotation_k = desired_rotation{k};
    desired_angular_velocity_k = desired_angular_velocity(:, k);

    % Tracking errors follow the original implementation: actual - desired.
    position_error = current_position - desired_position_k;
    orientation_error = 0.5 * vee3( ...
        current_rotation * desired_rotation_k' - ...
        desired_rotation_k * current_rotation');

    % Outer-loop commanded task-space velocity.
    linear_velocity_command = ...
        desired_linear_velocity_k - position_gain * position_error;
    angular_velocity_command = ...
        desired_angular_velocity_k - orientation_gain * orientation_error;
    task_velocity_command = [ ...
        linear_velocity_command; angular_velocity_command];

    % Joint-velocity box constraints.
    [lower_bound, upper_bound] = compute_velocity_bounds( ...
        joint_position, joint_position_min, joint_position_max, ...
        joint_velocity_limit, use_dynamic_joint_margin, joint_margin_gain);

    % Inner DNN iterations.
    for inner_idx = 1:num_inner_steps
        joint_velocity_output = project_box( ...
            vartheta, lower_bound, upper_bound);
        task_velocity_actual = task_jacobian * joint_velocity_output;

        y_dot = dnn_nu * ( ...
            task_velocity_command - ...
            task_jacobian * task_jacobian' * y_aux);

        vartheta_dot = project_box( ...
            dnn_mu * task_jacobian' * ...
            (task_velocity_command - task_velocity_actual) + ...
            task_jacobian' * y_aux, ...
            lower_bound, upper_bound);

        % Forward-Euler integration of the DNN states.
        y_aux = y_aux + inner_step * y_dot;
        vartheta = vartheta + inner_step * vartheta_dot;

        % Keep the DNN state within the admissible velocity box.
        vartheta = project_box(vartheta, lower_bound, upper_bound);
    end

    % Apply the projected joint velocity and update the joint position.
    joint_velocity_command = project_box( ...
        vartheta, lower_bound, upper_bound);
    joint_position = joint_position + joint_velocity_command * dt;
    joint_position = min( ...
        max(joint_position, joint_position_min), joint_position_max);

    % Store simulation data.
    actual_position_log(:, k) = current_position;
    actual_euler_log(:, k) = rotm_to_eulZYX(current_rotation);

    position_error_log(:, k) = position_error;
    orientation_error_log(:, k) = orientation_error;

    joint_position_log(:, k) = joint_position;
    joint_velocity_log(:, k) = joint_velocity_command;

    task_velocity_command_log(:, k) = task_velocity_command;
    task_velocity_actual_log(:, k) = ...
        task_jacobian * joint_velocity_command;
end

%% Backward-compatible output aliases and optional data export
ep_log = position_error_log;
eo_log = orientation_error_log;
traj_p_actual = actual_position_log;
traj_eul_act = actual_euler_log;
q_log = joint_position_log;
qdot_log = joint_velocity_log;

%% Tracking-error metrics
position_error_norm = sqrt(sum(position_error_log.^2, 1));
orientation_error_norm = sqrt(sum(orientation_error_log.^2, 1));

fprintf('\n================ Scheme 2: Tracking Results ================\n');
fprintf('Max position-error component    = %.6f m\n', ...
    max(abs(position_error_log(:))));
fprintf('RMS position-error norm         = %.6f m\n', ...
    sqrt(mean(position_error_norm.^2)));
fprintf('Max orientation-error component = %.6f rad\n', ...
    max(abs(orientation_error_log(:))));
fprintf('RMS orientation-error norm      = %.6f rad\n', ...
    sqrt(mean(orientation_error_norm.^2)));
fprintf('Max joint speed                 = %.6f rad/s\n', ...
    max(abs(joint_velocity_log(:))));
fprintf('=============================================================\n\n');

%% Plots
blue = [0.224, 0.325, 0.643];
red = [0.933, 0.118, 0.145];
green = [0.412, 0.741, 0.271];
plot_colors = {blue, red, green};
line_styles = {'-', ':', '--'};

% Position-tracking error.
figure('Name', 'Position tracking error', ...
    'Position', [100, 100, 1120, 240]);
hold on;
for i = 1:3
    plot(time, position_error_log(i, :), ...
        'Color', plot_colors{i}, ...
        'LineStyle', line_styles{i}, ...
        'LineWidth', 2.5);
end
ylabel('$\mathbf{e_p}$ [m]', 'Interpreter', 'latex');
legend({'$\mathbf{e_p}_x$', '$\mathbf{e_p}_y$', '$\mathbf{e_p}_z$'}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'Orientation', 'horizontal');
ax = gca;
set(ax, 'FontSize', 10, 'XColor', 'k', 'LineWidth', 1.2);

% Orientation-tracking error.
figure('Name', 'Orientation tracking error', ...
    'Position', [100, 100, 1120, 240]);
hold on;
for i = 1:3
    plot(time, orientation_error_log(i, :), ...
        'Color', plot_colors{i}, ...
        'LineStyle', line_styles{i}, ...
        'LineWidth', 2.5);
end
xlabel('t [s]');
ylabel('$\mathbf{e_o}$ [rad]', 'Interpreter', 'latex');
legend({'$\mathbf{e_o}_x$', '$\mathbf{e_o}_y$', '$\mathbf{e_o}_z$'}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'Orientation', 'horizontal');
ax = gca;
set(ax, 'FontSize', 10, 'XColor', 'k', 'LineWidth', 1.2);

% Desired and actual Euler angles.
figure('Name', 'Euler angles');
plot(time, desired_euler_log(1, :), 'LineWidth', 1.8); hold on;
plot(time, desired_euler_log(2, :), 'LineWidth', 1.8);
plot(time, desired_euler_log(3, :), 'LineWidth', 1.8);
plot(time, actual_euler_log(1, :), ':', 'LineWidth', 1.8);
plot(time, actual_euler_log(2, :), ':', 'LineWidth', 1.8);
plot(time, actual_euler_log(3, :), ':', 'LineWidth', 1.8);
xlabel('t [s]');
ylabel('Euler angle [rad]');
legend('\psi_d', '\theta_d', '\phi_d', ...
       '\psi_a', '\theta_a', '\phi_a', 'Location', 'best');
grid on;

% Euler-angle errors.
figure('Name', 'Euler angle errors');
plot(time, desired_euler_log - actual_euler_log, 'LineWidth', 1.8);
xlabel('t [s]');
ylabel('Euler error [rad]');
legend('\psi error', '\theta error', '\phi error', 'Location', 'best');
grid on;

% Three-dimensional position trajectory.
figure('Name', '3D trajectory');
plot3(actual_position_log(1, :), actual_position_log(2, :), ...
    actual_position_log(3, :), 'LineWidth', 2.0); hold on;
plot3(desired_position_log(1, :), desired_position_log(2, :), ...
    desired_position_log(3, :), 'r:', 'LineWidth', 2.0);
xlabel('x'); ylabel('y'); zlabel('z');
legend('Actual', 'Desired', 'Location', 'best');
grid on; box on;
view(-45, 25);

% Joint positions and velocities.
figure('Name', 'Joint angles and velocities');
subplot(2, 1, 1);
plot(time, joint_position_log', 'LineWidth', 1.5);
ylabel('q [rad]');
grid on;
legend('q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'Location', 'best');

subplot(2, 1, 2);
plot(time, joint_velocity_log', 'LineWidth', 1.5);
xlabel('t [s]');
ylabel('qdot [rad/s]');
grid on;
legend('qdot_1', 'qdot_2', 'qdot_3', 'qdot_4', 'qdot_5', 'qdot_6', ...
    'Location', 'best');

%% Local functions
function [lower_bound, upper_bound] = compute_velocity_bounds( ...
    joint_position, joint_position_min, joint_position_max, ...
    joint_velocity_limit, use_dynamic_joint_margin, joint_margin_gain)
%COMPUTE_VELOCITY_BOUNDS Construct joint-velocity box constraints.

num_joints = numel(joint_position);
upper_bound = joint_velocity_limit * ones(num_joints, 1);
lower_bound = -joint_velocity_limit * ones(num_joints, 1);

if use_dynamic_joint_margin
    upper_margin = joint_margin_gain * ...
        (joint_position_max - joint_position);
    lower_margin = joint_margin_gain * ...
        (joint_position_min - joint_position);

    upper_margin = max(upper_margin, zeros(num_joints, 1));
    lower_margin = min(lower_margin, zeros(num_joints, 1));

    upper_bound = min(upper_bound, upper_margin);
    lower_bound = max(lower_bound, lower_margin);
end

% Numerical safeguard: enforce upper_bound >= lower_bound.
invalid_idx = upper_bound < lower_bound;
midpoint = 0.5 * (upper_bound(invalid_idx) + lower_bound(invalid_idx));
upper_bound(invalid_idx) = midpoint;
lower_bound(invalid_idx) = midpoint;
end

function projected_value = project_box(value, lower_bound, upper_bound)
%PROJECT_BOX Elementwise Euclidean projection onto a box.
projected_value = min(max(value, lower_bound), upper_bound);
end

function vector = vee3(skew_matrix)
%VEE3 Map a 3-by-3 skew-symmetric matrix to a 3-vector.
vector = [skew_matrix(3, 2); skew_matrix(1, 3); skew_matrix(2, 1)];
end

function euler_angles = rotm_to_eulZYX(rotation_matrix)
%ROTM_TO_EULZYX Convert a rotation matrix to [yaw; pitch; roll].

pitch = atan2(-rotation_matrix(3, 1), ...
    sqrt(rotation_matrix(1, 1)^2 + rotation_matrix(2, 1)^2));

if abs(cos(pitch)) > 1e-8
    yaw = atan2(rotation_matrix(2, 1), rotation_matrix(1, 1));
    roll = atan2(rotation_matrix(3, 2), rotation_matrix(3, 3));
else
    % Graceful fallback near the ZYX Euler-angle singularity.
    yaw = atan2(-rotation_matrix(1, 2), rotation_matrix(2, 2));
    roll = 0;
end

euler_angles = [yaw; pitch; roll];
end
