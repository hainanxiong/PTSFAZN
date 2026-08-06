%% LZN Baseline: Pose Tracking With Online Jacobian Estimation
% This script evaluates an LZN-based pose-tracking controller on a PUMA 560
% manipulator. The task combines a multi-loop three-dimensional position
% trajectory with a time-varying axis-angle orientation trajectory. The
% Jacobian used by the controller is updated online.
%
% Requirements:
%   1. MATLAB.
%   2. Peter Corke's Robotics Toolbox for MATLAB (mdl_puma560, fkine,
%      jacob0, and ikcon).
%   3. Axis-angle conversion functions axang2rotm and rotm2axang.
%
% Reproducibility notes:
%   1. All trajectory, controller, neural-dynamics, projection, and
%      Jacobian-estimation parameters are defined explicitly below.
%   2. The supplied implementation uses a linear error activation for this
%      LZN baseline.
%   3. The exact Jacobian is used to generate task-space velocity data for
%      the online Jacobian estimator and for evaluation; the controller
%      matrix Q uses the current estimate J_h.

clear; clc; close all;

mdl_puma560;
robot = p560;

num_joints = robot.n;
position_dimension = 3;
orientation_dimension = 3;

sim_time = 20;             % Total simulation time [s]
dt = 0.005;                % Sampling interval [s]
time = 0:dt:sim_time;
num_samples = numel(time);

trajectory_radius = 0.03;
position_center = [0.5, 0.3, 0.2];

trajectory_phase = (0.5*pi/sim_time) * time;
sin_phase = sin(trajectory_phase);
cos_phase = cos(trajectory_phase);
sin_phase_squared = sin_phase.^2;

phase_6 = 6*pi * sin_phase_squared;
phase_4 = 4*pi * sin_phase_squared;
common_derivative = (pi/sim_time) * sin_phase .* cos_phase;
phase_6_dot = 6*pi * common_derivative;
phase_4_dot = 4*pi * common_derivative;

desired_position = [ ...
    position_center(1) + 2*trajectory_radius*sin(phase_6) ...
        - trajectory_radius*sin(phase_4) - 4*trajectory_radius; ...
    position_center(2) - 2*trajectory_radius*cos(phase_6) ...
        - trajectory_radius*cos(phase_4); ...
    position_center(3) + (2/3)*trajectory_radius*cos(phase_6) ...
        + (1/3)*trajectory_radius*cos(phase_4) - trajectory_radius];

desired_linear_velocity = [ ...
     2*trajectory_radius*cos(phase_6).*phase_6_dot ...
        - trajectory_radius*cos(phase_4).*phase_4_dot; ...
     2*trajectory_radius*sin(phase_6).*phase_6_dot ...
        + trajectory_radius*sin(phase_4).*phase_4_dot; ...
    -(2/3)*trajectory_radius*sin(phase_6).*phase_6_dot ...
        - (1/3)*trajectory_radius*sin(phase_4).*phase_4_dot];

rotation_axis = [-0.6405, 0.7634, -0.0842];
orientation_angle_variation = 0.31*sin(2*pi*time/15);
desired_rotation_angle = 2.6415 + orientation_angle_variation;

desired_rotation = cell(num_samples, 1);
desired_axis_angle = zeros(4, num_samples);

for k = 1:num_samples
    axis_angle_k = [rotation_axis, desired_rotation_angle(k)];
    desired_rotation{k} = axang2rotm(axis_angle_k);

    % Store as [axis_x; axis_y; axis_z; angle].
    axis_angle_from_rotation = rotm2axang(desired_rotation{k});
    desired_axis_angle(:, k) = axis_angle_from_rotation(:);
end

unit_rotation_axis = rotation_axis / norm(rotation_axis);

angular_velocity_magnitude = ...
    0.18 * 2*pi/15 * cos(2*pi*time/15);
desired_body_angular_velocity = zeros(3, num_samples);

for k = 1:num_samples
    desired_body_angular_velocity(:, k) = ...
        (angular_velocity_magnitude(k) .* unit_rotation_axis)';
end

initial_end_effector_position = [0.02; 0.001; 0.01];

initial_pose = eye(4);
initial_pose(1:3, 1:3) = desired_rotation{1};
initial_pose(1:3, 4) = initial_end_effector_position;
joint_position = robot.ikcon(initial_pose)';

J_h = robot.jacob0(joint_position);
Jp = J_h(1:3, :);
Jo = J_h(4:6, :);

eta = 0.1;                 % Neural-dynamics update gain
epsTheta = 1;              % Regularization: Theta = epsTheta*I

joint_velocity_limit = 3;  % Box constraint [rad/s]
lower_bound = -joint_velocity_limit * ones(num_joints, 1);
upper_bound = joint_velocity_limit * ones(num_joints, 1);

xi1 = 0.5;                 % Position-error feedback gain
xi2 = 0.6;                 % Orientation-error feedback gain
zeta = 1;                  % Jacobian-estimation gain

num_inner_iterations = 600;

% State: kappa = [joint_velocity; lambda_position; lambda_orientation].
kappa = zeros(num_joints + position_dimension + orientation_dimension, 1);
kappa(1:num_joints) = 0.2 * ones(num_joints, 1);

position_error_log = zeros(3, num_samples);
orientation_error_log = zeros(3, num_samples);

actual_position_log = zeros(3, num_samples);
desired_position_log = desired_position;

actual_axis_angle = zeros(4, num_samples);

joint_position_log = zeros(num_joints, num_samples);
joint_velocity_log = zeros(num_joints, num_samples);

estimated_jacobian_entry_log = zeros(1, num_samples);
exact_jacobian_entry_log = zeros(1, num_samples);

% Joint positions used only for the manipulator snapshot visualization.
joint_cartesian_log = zeros(3, num_joints, num_samples);

previous_task_velocity = 0.01 * ones(6, 1);
previous_joint_velocity = 0.01 * ones(num_joints, 1);

%% Main simulation loop
for k = 1:num_samples
    
    current_pose = robot.fkine(joint_position).T;
    [~, all_link_poses] = robot.fkine(joint_position);
    current_rotation = current_pose(1:3, 1:3);
    current_position = current_pose(1:3, 4);
    exact_jacobian = robot.jacob0(joint_position);

    for joint_idx = 1:num_joints
        joint_position_xyz = all_link_poses(joint_idx).transl;
        joint_cartesian_log(:, joint_idx, k) = joint_position_xyz(:);
    end

    current_axis_angle = rotm2axang(current_rotation);
    actual_axis_angle(:, k) = current_axis_angle(:);

   
    desired_position_k = desired_position(:, k);
    desired_linear_velocity_k = desired_linear_velocity(:, k);
    desired_rotation_k = desired_rotation{k};

    desired_body_omega_k = desired_body_angular_velocity(:, k);
    desired_spatial_omega_k = ...
        desired_rotation_k * desired_body_omega_k;

    position_error = current_position - desired_position_k;

    % Log-map orientation error used for evaluation.
    orientation_error = log_SO3_vec( ...
        desired_rotation_k' * current_rotation);

    % Skew-symmetric orientation error used by the LZN controller.
    orientation_error_controller = 0.5 * vee3( ...
        current_rotation * desired_rotation_k' - ...
        desired_rotation_k * current_rotation');

    % Linear activation used by the LZN baseline.
    activated_position_error = position_error;
    activated_orientation_error = orientation_error_controller;

    
    Theta = epsTheta * eye(num_joints);
    Q = [ ...
         Theta, Jp', Jo'; ...
        -Jp, zeros(position_dimension), ...
            zeros(position_dimension, orientation_dimension); ...
        -Jo, zeros(orientation_dimension, position_dimension), ...
            zeros(orientation_dimension)];

    joint_regularization = 10 * ones(num_joints, 1);
    B = [ ...
        joint_regularization; ...
        desired_linear_velocity_k - xi1*activated_position_error; ...
        desired_spatial_omega_k - xi2*activated_orientation_error];

    
    for inner_idx = 1:num_inner_iterations
        projected_argument = kappa - (Q*kappa + B);
        projected_argument = project_Xi( ...
            projected_argument, lower_bound, upper_bound, num_joints);
        kappa = kappa + eta * (-kappa + projected_argument);
    end

    
    joint_velocity = kappa(1:num_joints);
    joint_position = joint_position + joint_velocity * dt;

    joint_acceleration = ...
        (joint_velocity - previous_joint_velocity) / dt;
    previous_joint_velocity = joint_velocity;

    % The exact Jacobian is used here to generate task-space velocity data
    % for the estimator in simulation, consistent with the supplied script.
    true_jacobian = robot.jacob0(joint_position);
    task_velocity = true_jacobian * joint_velocity;
    task_acceleration = (task_velocity - previous_task_velocity) / dt;
    previous_task_velocity = task_velocity;

    J_h_dot = ( ...
        task_acceleration - J_h*joint_acceleration + ...
        zeta*(task_velocity - J_h*joint_velocity)) * joint_velocity' / ...
        (joint_velocity'*joint_velocity + 1e-6);

    J_h = J_h + J_h_dot * dt;
    Jp = J_h(1:3, :);
    Jo = J_h(4:6, :);

    position_error_log(:, k) = position_error;
    orientation_error_log(:, k) = orientation_error;

    actual_position_log(:, k) = current_position;
    joint_position_log(:, k) = joint_position;
    joint_velocity_log(:, k) = joint_velocity;

    estimated_jacobian_entry_log(k) = J_h(1, 2);
    exact_jacobian_entry_log(k) = exact_jacobian(1, 2);
end

ep_log = position_error_log;
eo_log = orientation_error_log;
traj_a = actual_position_log;
traj_d = desired_position_log;
traj_axis_angle_d = desired_axis_angle;
traj_axis_angle_a = actual_axis_angle;
q_trajectory = joint_position_log;
q_dot_trajectory = joint_velocity_log;
J11_a = estimated_jacobian_entry_log;
J11_d = exact_jacobian_entry_log;


LZN_P_rmse = sqrt(mean(ep_log.^2, 2));
LZN_O_rmse = sqrt(mean(eo_log.^2, 2));
LZN_P_mae = mean(abs(ep_log), 2);

fprintf('\n=================== LZN Tracking Summary ===================\n');
fprintf('Position RMSE [x y z] (m):       %.6f  %.6f  %.6f\n', ...
    LZN_P_rmse(1), LZN_P_rmse(2), LZN_P_rmse(3));
fprintf('Orientation RMSE [x y z] (rad):  %.6f  %.6f  %.6f\n', ...
    LZN_O_rmse(1), LZN_O_rmse(2), LZN_O_rmse(3));
fprintf('Position MAE [x y z] (m):        %.6f  %.6f  %.6f\n', ...
    LZN_P_mae(1), LZN_P_mae(2), LZN_P_mae(3));
fprintf('=============================================================\n');

%% Plots
% Manipulator snapshots along the desired and actual trajectories.
figure('Name', 'Manipulator motion snapshots');
hold on;
plot3(actual_position_log(1, :), actual_position_log(2, :), ...
    actual_position_log(3, :), '--', 'Color', [1, 0, 0], 'LineWidth', 2.5);
plot3(desired_position_log(1, :), desired_position_log(2, :), ...
    desired_position_log(3, :), 'Color', [0, 0, 1], 'LineWidth', 2.5);
grid on;
line([0, 0], [0, 0], [-0.2, 0], 'Color', 'k', 'LineWidth', 2.5);

link_colors = {'y', 'c', 'm', 'g', 'r', 'b'};
for sample_idx = 1:10:num_samples
    for joint_idx = 1:num_joints - 1
        p1 = joint_cartesian_log(:, joint_idx, sample_idx);
        p2 = joint_cartesian_log(:, joint_idx + 1, sample_idx);
        line([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], ...
            'Color', link_colors{joint_idx});
    end

    p6 = joint_cartesian_log(:, num_joints, sample_idx);
    pe = actual_position_log(:, sample_idx);
    line([p6(1), pe(1)], [p6(2), pe(2)], [p6(3), pe(3)], ...
        'Color', link_colors{num_joints});
end

ax = gca;
set(ax, 'FontSize', 10, 'XColor', 'k', 'LineWidth', 1.2);
box on;

% Desired and actual end-effector position trajectories.
figure('Name', 'Desired and actual trajectories');
hold on;
plot3(desired_position_log(1, :), desired_position_log(2, :), ...
    desired_position_log(3, :), 'LineWidth', 2.5);
plot3(actual_position_log(1, :), actual_position_log(2, :), ...
    actual_position_log(3, :), '--', 'LineWidth', 2.5, ...
    'Marker', 's', 'MarkerIndices', 1:100:num_samples, ...
    'MarkerSize', 6, 'MarkerFaceColor', 'w');
grid on; box off;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
legend('Desired path', 'Actual path', 'Location', 'north');
ax = gca;
set(ax, 'FontSize', 10, 'XColor', 'k', 'LineWidth', 1.2);
view(-45, 25);

% Position errors and Euclidean norm.
figure('Name', 'Position tracking error');
subplot(2, 1, 1);
line_styles = {'-.', ':', '--'};
hold on;
for i = 1:3
    plot(time, position_error_log(i, :), ...
        'LineStyle', line_styles{i}, 'LineWidth', 2.5);
end
ylabel('$\mathbf{e_p}$ [m]', 'Interpreter', 'latex');
legend({'$\mathbf{e_p}_x$', '$\mathbf{e_p}_y$', '$\mathbf{e_p}_z$'}, ...
    'Interpreter', 'latex', 'FontSize', 14, ...
    'Location', 'best', 'Orientation', 'horizontal');
format_error_axes(gca, 1.5);

subplot(2, 1, 2);
plot(time, vecnorm(position_error_log, 2, 1)', ...
    'Color', [0.4940, 0.1840, 0.5560], 'LineWidth', 2.5);
xlabel('t [s]');
ylabel('$||\mathbf{e_p}||_2$ [m]', 'Interpreter', 'latex');
legend('$||\mathbf{e_p}||_2$', 'Interpreter', 'latex', ...
    'FontSize', 14, 'Location', 'north', 'Orientation', 'horizontal');
format_error_axes(gca, 1.5);

% Orientation errors and Euclidean norm.
figure('Name', 'Orientation tracking error');
subplot(2, 1, 1);
hold on;
for i = 1:3
    plot(time, orientation_error_log(i, :), ...
        'LineStyle', line_styles{i}, 'LineWidth', 2.5);
end
ylabel('$\mathbf{e_o}$ [rad]', 'Interpreter', 'latex');
legend({'$\mathbf{e_o}_x$', '$\mathbf{e_o}_y$', '$\mathbf{e_o}_z$'}, ...
    'Interpreter', 'latex', 'FontSize', 14, ...
    'Location', 'best', 'Orientation', 'horizontal');
format_error_axes(gca, 2.0);

subplot(2, 1, 2);
plot(time, vecnorm(orientation_error_log, 2, 1)', ...
    'Color', [0.4940, 0.1840, 0.5560], 'LineWidth', 2.5);
xlabel('t [s]');
ylabel('$||\mathbf{e_o}||_2$ [rad]', 'Interpreter', 'latex');
legend('$||\mathbf{e_o}||_2$', 'Interpreter', 'latex', ...
    'FontSize', 14, 'Location', 'north', 'Orientation', 'horizontal');
format_error_axes(gca, 1.5);

%% Local functions
function vector = vee3(skew_matrix)
%VEE3 Map a 3-by-3 skew-symmetric matrix to a 3-vector.
vector = [skew_matrix(3, 2); skew_matrix(1, 3); skew_matrix(2, 1)];
end

function vector = log_SO3_vec(rotation_matrix)
%LOG_SO3_VEC Logarithmic-map vector of an SO(3) rotation matrix.
cos_theta = (trace(rotation_matrix) - 1) / 2;
cos_theta = max(min(cos_theta, 1), -1);
theta = acos(cos_theta);

if theta < 1e-9
    vector = [0; 0; 0];
else
    vector = theta/(2*sin(theta)) * vee3( ...
        rotation_matrix - rotation_matrix');
end
end

function projected_value = project_Xi( ...
    value, lower_bound, upper_bound, num_joints)
%PROJECT_XI Project joint-velocity states onto Xi; leave dual states free.
projected_value = value;
projected_value(1:num_joints) = min( ...
    max(value(1:num_joints), lower_bound), upper_bound);
end

function format_error_axes(ax, line_width)
%FORMAT_ERROR_AXES Apply the common publication plotting style.
set(ax, 'FontSize', 10, 'XColor', 'k', 'LineWidth', line_width);
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';
ax.XMinorTick = 'off';
ax.YMinorTick = 'off';
ax.GridLineStyle = ':';
ax.GridColor = [0, 0, 0];
ax.GridAlpha = 0.5;
end
