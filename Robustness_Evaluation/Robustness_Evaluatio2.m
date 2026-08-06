%% Scheme 2: Data-Based Model-Free Predictive Pose Control
% Comparison implementation based on:
% M. Cao, "Data-based model-free predictive control system under the
% design philosophy of MPC and zeroing neurodynamics for robotic arm pose
% tracking."
%
% This script evaluates pose tracking of a PUMA 560 manipulator using a
% data-based model-free predictive control formulation combined with a
% zeroing-neurodynamics (ZNN) solver and online Jacobian estimation. This
% version retains the D-H-parameter perturbation block from the original
% comparison script.
%
% Requirements:
%   1. MATLAB.
%   2. Peter Corke's Robotics Toolbox for MATLAB (mdl_puma560, SerialLink,
%      fkine, jacob0, and ikcon).
%
% Reproducibility notes:
%   1. All simulation, MPC, ZNN, constraint, perturbation, and
%      Jacobian-estimation parameters are defined explicitly below.
%   2. hat_J is the Jacobian supplied to the controller. J_true is computed
%      only for monitoring and is not used by the MPC/ZNN control law.
%   3. Pose vectors follow [x; y; z; yaw; pitch; roll], with ZYX Euler
%      angles expressed in radians.

clear; clc; close all;

mdl_puma560;
robot = p560;

num_joints = robot.n;
task_dimension = 6;

sim_time = 20;             % Total simulation time [s]
sampling_time = 0.005;     % Sampling interval [s]
num_steps = round(sim_time / sampling_time);
time = 0:sampling_time:sim_time;

prediction_horizon = 5;
control_horizon = 5;

% Joint position limits from the PUMA 560 model.
joint_position_min = robot.qlim(:, 1);
joint_position_max = robot.qlim(:, 2);

% Joint velocity and acceleration limits.
joint_velocity_limit = deg2rad([90; 90; 90; 120; 120; 180]);
joint_velocity_max = joint_velocity_limit;
joint_velocity_min = -joint_velocity_limit;

joint_acceleration_max = [20; 20; 20; 20; 20; 20];
joint_acceleration_min = -joint_acceleration_max;

% Horizon-wise constraint matrices.
Theta_max = repmat(joint_position_max', prediction_horizon, 1);
Theta_min = repmat(joint_position_min', prediction_horizon, 1);
Theta_dot_max = repmat(joint_velocity_max', control_horizon, 1);
Theta_dot_min = repmat(joint_velocity_min', control_horizon, 1);
Theta_ddot_max = repmat(joint_acceleration_max', control_horizon, 1);
Theta_ddot_min = repmat(joint_acceleration_min', control_horizon, 1);

% MPC weighting matrices.
Q1_prime = diag([1.0e5, 1.0e5, 1.0e5, 2.0e3, 2.0e3, 2.0e4]);
Q2_prime = 4.0 * eye(num_joints);
Q3_prime = 40.0 * eye(num_joints);

Q1 = kron(eye(prediction_horizon), Q1_prime);
Q2 = kron(eye(control_horizon), Q2_prime);
Q3 = kron(eye(control_horizon), Q3_prime);

% Prediction matrix C.
C = zeros(prediction_horizon, control_horizon);
for i = 1:prediction_horizon
    for j = 1:control_horizon
        if i >= j
            C(i, j) = (i - j + 1) * sampling_time;
        end
    end
end

% Lower-triangular accumulation matrix and inequality matrix E for E*v <= c.
tilde_I = tril(ones(control_horizon));
I_Nc = eye(control_horizon);

E_prime = [ ...
    C; -C; ...
    tilde_I; -tilde_I; ...
    I_Nc; -I_Nc];
E = kron(eye(num_joints), E_prime);
I_kron_tildeI = kron(eye(num_joints), tilde_I);

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
     3*trajectory_scale*trajectory_omega .* sin_phase.^2 .* cos_phase]; %#ok<NASGU>

yaw_offset = 1.12;
pitch_offset = 0.65;
roll_offset = 0.10;

desired_yaw = 0.05*cos(time) + yaw_offset;
desired_pitch = -0.032*sin(time) + pitch_offset;
desired_roll = 0.015*sin(time) + roll_offset;

desired_poses = [ ...
    desired_position; desired_yaw; desired_pitch; desired_roll];

initial_position_error = [0.0082; -0.0031; -0.0084];

initial_pose = eye(4);
initial_pose(1:3, 1:3) = eulZYX_to_rotm( ...
    desired_yaw(1), desired_pitch(1), desired_roll(1));
initial_pose(1:3, 4) = ...
    desired_position(:, 1) + initial_position_error;

initial_joint_seed = [0; -pi/4; pi/2; 0; pi/4; 0];
initial_joint_position = robot.ikcon(initial_pose, initial_joint_seed')';

joint_angles = zeros(num_joints, num_steps + 1);
joint_velocities = zeros(num_joints, num_steps + 1);
joint_accelerations = zeros(num_joints, num_steps + 1);

actual_poses = zeros(task_dimension, num_steps + 1);
pose_errors = zeros(task_dimension, num_steps + 1);

hat_J = zeros(task_dimension, num_joints, num_steps + 1);
J_true = zeros(task_dimension, num_joints, num_steps + 1);

% Low-pass-filter states used by the online Jacobian estimator.
s_f = zeros(task_dimension, num_steps + 1);
v_f = zeros(task_dimension, num_steps + 1);
ds_f = zeros(task_dimension, num_steps + 1);
dv_f = zeros(task_dimension, num_steps + 1);

joint_angles(:, 1) = initial_joint_position;
joint_velocities(:, 1) = zeros(num_joints, 1);
joint_accelerations(:, 1) = zeros(num_joints, 1);

actual_poses(:, 1) = get_pose_ypr(robot, initial_joint_position);
pose_errors(:, 1) = actual_poses(:, 1) - desired_poses(:, 1);

s_f(:, 1) = actual_poses(:, 1);
v_f(:, 1) = zeros(task_dimension, 1);

hat_J(:, :, 1) = analytical_pose_jacobian(robot, initial_joint_position);
J_true(:, :, 1) = hat_J(:, :, 1);

lambda_zn = 1;             % ZNN convergence-rate parameter
max_iterations = 10;
solver_tolerance = 1e-8;
delta_regularization = 1e-8;

tau1 = 1/(200*pi);
tau2 = 1/(200*pi);
eta = 10;                  % DZN-JMA design parameter
pseudoinverse_regularization = 0.5;

%% Main simulation loop
for k = 1:num_steps
    current_time = time(k);
    dh_parameters = p560.links;

    perturbation_a2 = 0.003;
    perturbation_d3 = 0.001 + 0.01*cos(0.025*current_time);
    perturbation_a3 = 0.001 + 0.03*cos(0.015*current_time);
    perturbation_d4 = 0.015;

    dh_parameters(2).a = 0.4318 + perturbation_a2;
    dh_parameters(3).d = 0.15005 + perturbation_d3;
    dh_parameters(3).a = 0.0203 + perturbation_a3;
    dh_parameters(4).d = 0.4318 + perturbation_d4;

    % Reconstruct the modified PUMA 560 object exactly as in the original
    % implementation. In Peter Corke's Robotics Toolbox, Link objects are
    % reference objects; therefore, the modified link parameters are shared
    % with 'robot' and are reflected in its subsequent kinematic evaluation.
    p560 = SerialLink(dh_parameters, 'name', 'Modified Puma 560');

    joint_position_k = joint_angles(:, k);
    estimated_jacobian_k = hat_J(:, :, k);

    %% 10.2 Current pose and monitoring quantities
    actual_poses(:, k) = get_pose_ypr(robot, joint_position_k);
    pose_errors(:, k) = pose_difference( ...
        actual_poses(:, k), desired_poses(:, k));

    % The exact task Jacobian is recorded only for monitoring/evaluation.
    J_true(:, :, k) = analytical_pose_jacobian(robot, joint_position_k);

    J_kron_C = kron(estimated_jacobian_k, C);
    M = 2 * ( ...
        J_kron_C' * Q1 * J_kron_C + ...
        I_kron_tildeI' * Q2 * I_kron_tildeI + Q3);

    H = repmat(actual_poses(:, k)', prediction_horizon, 1);
    S = zeros(prediction_horizon, task_dimension);
    for i = 1:prediction_horizon
        reference_idx = min(k + i, num_steps + 1);
        S(i, :) = desired_poses(:, reference_idx)';
    end

    if k == 1
        baseline_joint_velocity = joint_velocities(:, k);
    else
        baseline_joint_velocity = joint_velocities(:, k - 1);
    end

    B = zeros(prediction_horizon, num_joints);
    for i = 1:prediction_horizon
        B(i, :) = i * sampling_time * baseline_joint_velocity';
    end
    A = repmat(baseline_joint_velocity', control_horizon, 1);

    % Vectorized MPC terms. MATLAB column-major vectorization is used
    % consistently with the Kronecker-product construction above.
    vec_H = reshape(H, [], 1);
    vec_B_JT = reshape(B * estimated_jacobian_k', [], 1);
    vec_S = reshape(S, [], 1);
    vec_A = reshape(A, [], 1);

    m_bf = 2 * J_kron_C' * Q1 * (vec_H + vec_B_JT - vec_S) + ...
        2 * I_kron_tildeI' * Q2 * vec_A;

    % Inequality-constraint vector c for E*v <= c.
    D = repmat(joint_position_k', prediction_horizon, 1);
    F = [ ...
        Theta_max - D - B; ...
       -Theta_min + D + B; ...
        Theta_dot_max - A; ...
       -Theta_dot_min + A; ...
        sampling_time * Theta_ddot_max; ...
       -sampling_time * Theta_ddot_min];
    c = reshape(F, [], 1);

    v = zeros(num_joints * control_horizon, 1);
    rho = zeros(size(c));
    znn_state = [v; rho];

    K = [M, E'; -E, eye(size(E, 1))];

    w = c - E * v;
    d = sqrt(w.^2 + rho.^2 + delta_regularization);
    y = [m_bf; c - d];

    Z1 = diag(w ./ d);
    Z2 = diag(rho ./ d);
    W = [M, E'; -E + Z1 * E, eye(size(E, 1)) - Z2];

    for iteration = 1:max_iterations
        znn_error = K * znn_state + y;
        search_direction = W \ (lambda_zn * znn_error);
        znn_state = znn_state - search_direction;

        v = znn_state(1:num_joints * control_horizon);
        rho = znn_state(num_joints * control_horizon + 1:end);

        w = c - E * v;
        d = sqrt(w.^2 + rho.^2 + delta_regularization);
        y = [m_bf; c - d];
        Z1 = diag(w ./ d);
        Z2 = diag(rho ./ d);
        W = [M, E'; -E + Z1 * E, eye(size(E, 1)) - Z2];

        if norm(znn_error, 2) < solver_tolerance
            break;
        end
    end

    control_sequence = reshape(v, [control_horizon, num_joints]);
    velocity_increment = control_sequence(1, :)';

    if k == 1
        joint_velocities(:, k) = velocity_increment;
    else
        joint_velocities(:, k) = ...
            joint_velocities(:, k - 1) + velocity_increment;
    end

    % Enforce joint-velocity and joint-acceleration limits.
    joint_velocities(:, k) = min( ...
        max(joint_velocities(:, k), joint_velocity_min), joint_velocity_max);

    joint_accelerations(:, k) = velocity_increment / sampling_time;
    joint_accelerations(:, k) = min( ...
        max(joint_accelerations(:, k), joint_acceleration_min), ...
        joint_acceleration_max);

    % Update and clamp the joint positions.
    joint_angles(:, k + 1) = ...
        joint_position_k + joint_velocities(:, k) * sampling_time;
    joint_angles(:, k + 1) = min( ...
        max(joint_angles(:, k + 1), joint_position_min), joint_position_max);

    %% 10.6 Update the Jacobian estimate from filtered data
    s_f(:, k + 1) = ...
        (1 - sampling_time/(tau1 + sampling_time)) * s_f(:, k) + ...
        sampling_time/(tau1 + sampling_time) * actual_poses(:, k);
    ds_f(:, k) = (1/tau1) * pose_difference( ...
        actual_poses(:, k), s_f(:, k + 1));

    v_f(:, k + 1) = ...
        (1 - sampling_time/(tau2 + sampling_time)) * v_f(:, k) + ...
        sampling_time/(tau2 + sampling_time) * ds_f(:, k);
    dv_f(:, k) = (1/tau2) * (ds_f(:, k) - v_f(:, k + 1));

    joint_velocity_k = joint_velocities(:, k);
    joint_acceleration_k = joint_accelerations(:, k);

    joint_velocity_pseudoinverse = joint_velocity_k' / ( ...
        joint_velocity_k' * joint_velocity_k + ...
        pseudoinverse_regularization);

    J_dot = ( ...
        dv_f(:, k) - estimated_jacobian_k * joint_acceleration_k + ...
        eta * (ds_f(:, k) - estimated_jacobian_k * joint_velocity_k)) * ...
        joint_velocity_pseudoinverse;

    hat_J(:, :, k + 1) = ...
        estimated_jacobian_k + sampling_time * J_dot;

    % Refresh the pose and tracking error at the next sample.
    actual_poses(:, k + 1) = get_pose_ypr(robot, joint_angles(:, k + 1));
    pose_errors(:, k + 1) = pose_difference( ...
        actual_poses(:, k + 1), desired_poses(:, k + 1));
end

% Exact Jacobian at the final sample, retained only for evaluation.
J_true(:, :, num_steps + 1) = analytical_pose_jacobian( ...
    robot, joint_angles(:, num_steps + 1));

%% 11. Tracking-error metrics and backward-compatible outputs
position_error = pose_errors(1:3, :);
ori_error = pose_errors(4:6, :);

% Preserve the original output variable names used by repository plotting
% and post-processing scripts.
ep_log = position_error;
eo_log = ori_error;

position_error_norm = sqrt(sum(position_error.^2, 1));
orientation_error_norm = sqrt(sum(ori_error.^2, 1));

fprintf('\n================ Scheme 2: Tracking Summary ================\n');
fprintf('Position RMS error    = %.6f m\n', ...
    sqrt(mean(position_error_norm.^2)));
fprintf('Position max error    = %.6f m\n', max(position_error_norm));
fprintf('Orientation RMS error = %.6f rad\n', ...
    sqrt(mean(orientation_error_norm.^2)));
fprintf('Orientation max error = %.6f rad\n', max(orientation_error_norm));
fprintf('Max joint speed       = %.6f rad/s\n', ...
    max(abs(joint_velocities(:))));
fprintf('=============================================================\n');

%% Plots
% Three-dimensional end-effector trajectory.
figure('Name', '3D trajectory');
plot3(actual_poses(1, :), actual_poses(2, :), actual_poses(3, :), ...
    'LineWidth', 2.0); hold on;
plot3(desired_poses(1, :), desired_poses(2, :), desired_poses(3, :), ...
    'r--', 'LineWidth', 2.0);
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('Actual', 'Desired', 'Location', 'best');
grid on; box on; view(-45, 25);
title('End-effector trajectory');

% Diagnostic position-error plot.
figure('Name', 'Position errors');
plot(time, position_error(1, :), 'LineWidth', 1.6); hold on;
plot(time, position_error(2, :), 'LineWidth', 1.6);
plot(time, position_error(3, :), 'LineWidth', 1.6);
xlabel('t [s]'); ylabel('Position error [m]');
legend('e_x', 'e_y', 'e_z', 'Location', 'best');
grid on;

% Diagnostic orientation-error plot.
figure('Name', 'Orientation errors (ZYX Euler)');
plot(time, ori_error(1, :), 'LineWidth', 1.6); hold on;
plot(time, ori_error(2, :), 'LineWidth', 1.6);
plot(time, ori_error(3, :), 'LineWidth', 1.6);
xlabel('t [s]'); ylabel('Orientation error [rad]');
legend('e_{yaw}', 'e_{pitch}', 'e_{roll}', 'Location', 'best');
grid on;

% Publication-style tracking-error plots.
blue = [0.224, 0.325, 0.643];
red = [0.933, 0.118, 0.145];
green = [0.412, 0.741, 0.271];
plot_colors = {blue, red, green};
line_styles = {'-', ':', '--'};

figure('Name', 'Position tracking error', ...
    'Position', [100, 100, 1120, 240]);
hold on;
for i = 1:3
    plot(time, position_error(i, :), ...
        'Color', plot_colors{i}, ...
        'LineStyle', line_styles{i}, ...
        'LineWidth', 2.5);
end
ylabel('$\mathbf{e_p}$ [m]', 'Interpreter', 'latex');
legend({'$\mathbf{e_p}_x$', '$\mathbf{e_p}_y$', '$\mathbf{e_p}_z$'}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'Orientation', 'horizontal');
ax = gca;
set(ax, 'FontSize', 10, 'XColor', 'k', 'LineWidth', 1.2);

figure('Name', 'Orientation tracking error', ...
    'Position', [100, 100, 1120, 240]);
hold on;
for i = 1:3
    plot(time, ori_error(i, :), ...
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
figure('Name', 'Desired vs. actual Euler angles');
subplot(3, 1, 1);
plot(time, desired_poses(4, :), 'r--', 'LineWidth', 1.6); hold on;
plot(time, actual_poses(4, :), 'b', 'LineWidth', 1.3);
ylabel('yaw [rad]');
legend('Desired', 'Actual', 'Location', 'best'); grid on;

subplot(3, 1, 2);
plot(time, desired_poses(5, :), 'r--', 'LineWidth', 1.6); hold on;
plot(time, actual_poses(5, :), 'b', 'LineWidth', 1.3);
ylabel('pitch [rad]');
legend('Desired', 'Actual', 'Location', 'best'); grid on;

subplot(3, 1, 3);
plot(time, desired_poses(6, :), 'r--', 'LineWidth', 1.6); hold on;
plot(time, actual_poses(6, :), 'b', 'LineWidth', 1.3);
ylabel('roll [rad]'); xlabel('t [s]');
legend('Desired', 'Actual', 'Location', 'best'); grid on;

% Joint positions and velocities.
figure('Name', 'Joint angles and velocities');
subplot(2, 1, 1);
plot(time(1:end-1), joint_angles(:, 1:end-1)', 'LineWidth', 1.2);
ylabel('q [rad]'); grid on;
legend('q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'Location', 'best');

subplot(2, 1, 2);
plot(time(1:end-1), joint_velocities(:, 1:end-1)', 'LineWidth', 1.2);
xlabel('t [s]'); ylabel('qdot [rad/s]'); grid on;
legend('qdot_1', 'qdot_2', 'qdot_3', 'qdot_4', 'qdot_5', 'qdot_6', ...
    'Location', 'best');

%% Local functions
function pose = get_pose_ypr(robot, joint_position)
%GET_POSE_YPR Return [x; y; z; yaw; pitch; roll].
current_pose = robot.fkine(joint_position).T;
position = current_pose(1:3, 4);
rotation = current_pose(1:3, 1:3);
ypr = rotm_to_eulZYX(rotation);
pose = [position; ypr];
end

function J_pose = analytical_pose_jacobian(robot, joint_position)
%ANALYTICAL_POSE_JACOBIAN Jacobian for [p_dot; yaw_dot; pitch_dot; roll_dot].
current_pose = robot.fkine(joint_position).T;
rotation = current_pose(1:3, 1:3);
ypr = rotm_to_eulZYX(rotation);

pitch = ypr(2);
roll = ypr(3);

geometric_jacobian = robot.jacob0(joint_position');
linear_jacobian = geometric_jacobian(1:3, :);
angular_jacobian = geometric_jacobian(4:6, :);

% omega_body = R' * omega_spatial
% [roll_dot; pitch_dot; yaw_dot] = inv(T_eul) * omega_body
% [yaw_dot; pitch_dot; roll_dot] = P * [roll_dot; pitch_dot; yaw_dot]
T_eul = [ ...
    1, 0, -sin(pitch); ...
    0, cos(roll),  sin(roll)*cos(pitch); ...
    0, -sin(roll), cos(roll)*cos(pitch)];

% Numerical safeguard near a ZYX Euler-angle singularity.
if abs(det(T_eul)) < 1e-8
    T_eul_inverse = pinv(T_eul);
else
    T_eul_inverse = inv(T_eul);
end

euler_order_map = [0, 0, 1; 0, 1, 0; 1, 0, 0];
euler_jacobian = ...
    euler_order_map * T_eul_inverse * rotation' * angular_jacobian;

J_pose = [linear_jacobian; euler_jacobian];
end

function pose_difference_value = pose_difference(pose_a, pose_b)
%POSE_DIFFERENCE Compute pose_a - pose_b with wrapped Euler-angle errors.
pose_difference_value = pose_a - pose_b;
pose_difference_value(4:6) = wrap_to_pi_local( ...
    pose_difference_value(4:6));
end

function angle = wrap_to_pi_local(angle)
%WRAP_TO_PI_LOCAL Wrap angles to [-pi, pi).
angle = mod(angle + pi, 2*pi) - pi;
end

function euler_angles = rotm_to_eulZYX(rotation_matrix)
%ROTM_TO_EULZYX Convert a rotation matrix to [yaw; pitch; roll].
pitch = atan2(-rotation_matrix(3, 1), ...
    sqrt(rotation_matrix(1, 1)^2 + rotation_matrix(2, 1)^2));

if abs(cos(pitch)) > 1e-8
    yaw = atan2(rotation_matrix(2, 1), rotation_matrix(1, 1));
    roll = atan2(rotation_matrix(3, 2), rotation_matrix(3, 3));
else
    yaw = atan2(-rotation_matrix(1, 2), rotation_matrix(2, 2));
    roll = 0;
end

euler_angles = [yaw; pitch; roll];
end

function rotation_matrix = eulZYX_to_rotm(yaw, pitch, roll)
%EULZYX_TO_ROTM Construct Rz(yaw)*Ry(pitch)*Rx(roll).
cy = cos(yaw);   sy = sin(yaw);
cp = cos(pitch); sp = sin(pitch);
cr = cos(roll);  sr = sin(roll);

rotation_matrix = [ ...
    cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr; ...
    sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr; ...
    -sp,   cp*sr,              cp*cr];
end
