%% Proposed Model-Free Pose Tracking Control with Online Jacobian Estimation
% This script simulates pose tracking control for a PUMA560 robotic manipulator.
% It includes desired path generation, online Jacobian estimation, controller
% execution, performance summary, and visualization.

clear; clc; close all;

mdl_puma560;
robot = p560;
numJoints = robot.n;

positionDim = 3;
orientationDim = 3;

Tfinal = 20;
dt = 0.005;
t = 0:dt:Tfinal;
numSteps = numel(t);

numPetals = 8;
baseRadius = 0.04;
petalGain = 0.45;
centerX = 0.30;
centerY = 0.00;
centerZ = 0.30;

tiltXDeg = 60;
tiltZDeg = 60;
slopeX = tan(deg2rad(tiltXDeg));
slopeZ = tan(deg2rad(tiltZDeg));

rho = baseRadius * (1 + petalGain * cos(numPetals * t));
rhoDot = -baseRadius * petalGain * numPetals * sin(numPetals * t);

xDesired = centerX + rho .* cos(t);
zDesired = centerZ + rho .* sin(t);
yDesired = centerY + slopeX * (xDesired - centerX) + slopeZ * (zDesired - centerZ);

xDotDesired = rhoDot .* cos(t) - rho .* sin(t);
zDotDesired = rhoDot .* sin(t) + rho .* cos(t);
yDotDesired = slopeX .* xDotDesired + slopeZ .* zDotDesired;

positionDesired = [xDesired; yDesired; zDesired];
velocityDesired = [xDotDesired; yDotDesired; zDotDesired];

alpha0 = 1.6;
beta0 = 0.65;
gamma0 = 0.1;

alphaDesired = 0.1 * sin(t) + alpha0;
betaDesired = -0.2 * sin(t) + beta0;
gammaDesired = 0.2 * sin(t) + gamma0;

alphaDot = 0.1 * cos(t);
betaDot = -0.2 * cos(t);
gammaDot = 0.2 * cos(t);

rotationDesired = cell(numSteps, 1);
bodyAngularVelocityDesired = zeros(3, numSteps);
spatialAngularVelocityDesired = zeros(3, numSteps);

for idx = 1:numSteps
    rotationDesired{idx} = eul2rotm([alphaDesired(idx), betaDesired(idx), gammaDesired(idx)], 'ZYX');
    RDesiredCurrent = rotationDesired{idx};

    roll = gammaDesired(idx);
    pitch = betaDesired(idx);
    yaw = alphaDesired(idx);

    rollDot = gammaDot(idx);
    pitchDot = betaDot(idx);
    yawDot = alphaDot(idx);

    eulerRateMap = [1, 0, -sin(pitch);
                    0, cos(roll), sin(roll) * cos(pitch);
                    0, -sin(roll), cos(roll) * cos(pitch)];

    bodyAngularVelocityDesired(:, idx) = eulerRateMap * [rollDot; pitchDot; yawDot];
    spatialAngularVelocityDesired(:, idx) = RDesiredCurrent * bodyAngularVelocityDesired(:, idx);
end

initialPositionError = [0.082; -0.031; -0.084];
initialRotation = rotationDesired{1} * eul2rotm([0.2, 0.1, 0.1], 'ZYX');
initialPosition = positionDesired(:, 1) + initialPositionError;

initialTransform = eye(4);
initialTransform(1:3, 1:3) = initialRotation;
initialTransform(1:3, 4) = initialPosition;

q = robot.ikcon(initialTransform)';
JEstimated = robot.jacob0(q);
Jp = JEstimated(1:3, :);
Jo = JEstimated(4:6, :);

solverStep = 0.1;
epsTheta = 1;
qdotLimit = 4;
lowerBound = -qdotLimit * ones(numJoints, 1);
upperBound = qdotLimit * ones(numJoints, 1);

rhoZN = 1;
xiPosition = 0.11;
xiOrientation = 0.2;
jacobianAdaptationGain = 10;

qdot = zeros(numJoints, 1);
qddot = zeros(numJoints, 1);
qdotPrevious = zeros(numJoints, 1);

kappa = zeros(numJoints + positionDim + orientationDim, 1);
kappa(1:numJoints) = 0.2 * ones(numJoints, 1);

positionErrorLog = zeros(3, numSteps);
orientationErrorLog = zeros(3, numSteps);
actualPositionLog = zeros(3, numSteps);
desiredPositionLog = zeros(3, numSteps);

desiredEulerLog = [alphaDesired; betaDesired; gammaDesired];
actualEulerLog = zeros(3, numSteps);

qLog = zeros(numJoints, numSteps);
qdotLog = zeros(numJoints, numSteps);
qddotLog = zeros(numJoints, numSteps);
actualRotationLog = cell(numSteps, 1);

jacobianElementSelector = 42;


tauPositionFilter = 1 / (200 * pi);
tauVelocityFilter = 1 / (200 * pi);
pinvRegularization = 0.5;

initialEndEffectorTransform = robot.fkine(q).T;
filteredPosition = initialEndEffectorTransform(1:3, 4);
previousRotationForJacobian = initialEndEffectorTransform(1:3, 1:3);
filteredEndEffectorVelocity = zeros(6, 1);

varsigmaPosition = 0.01;
varsigmaOrientation = 0.01;

positionErrorIntegral = zeros(3, 1);
orientationErrorIntegral = zeros(3, 1);

for idx = 1:numSteps
    currentTransform = robot.fkine(q).T;
    currentRotation = currentTransform(1:3, 1:3);
    currentPosition = currentTransform(1:3, 4);
    actualRotationLog{idx} = currentRotation;

    currentEuler = rotm2eul(currentRotation, 'ZYX');
    actualEulerLog(:, idx) = currentEuler(:);

    desiredPositionCurrent = positionDesired(:, idx);
    desiredVelocityCurrent = velocityDesired(:, idx);
    desiredRotationCurrent = rotationDesired{idx};
    desiredAngularVelocityCurrent = spatialAngularVelocityDesired(:, idx);

    positionError = currentPosition - desiredPositionCurrent;
    orientationError = 0.5 * vee(currentRotation * desiredRotationCurrent' - desiredRotationCurrent * currentRotation');

    positionErrorIntegral = positionErrorIntegral + positionError * dt;
    orientationErrorIntegral = orientationErrorIntegral + orientationError * dt;

    psiPosition = positionError + varsigmaPosition * positionErrorIntegral;
    psiOrientation = orientationError + varsigmaOrientation * orientationErrorIntegral;

    activatedPositionError = zdActivation(psiPosition, rhoZN);
    activatedOrientationError = zdActivation(psiOrientation, rhoZN);

    thetaMatrix = epsTheta * eye(numJoints);
    Q = [thetaMatrix, Jp', Jo';
         -Jp, zeros(positionDim), zeros(positionDim, orientationDim);
         -Jo, zeros(orientationDim, positionDim), zeros(orientationDim)];

    etaPosition = fuzzyErrorToEta(norm(psiPosition));
    etaOrientation = fuzzyErrorToEta(norm(psiOrientation));

    auxiliaryVector = 10 * ones(numJoints, 1);
    B = [auxiliaryVector;
         desiredVelocityCurrent - (xiPosition + etaPosition) * activatedPositionError;
         desiredAngularVelocityCurrent - (xiOrientation + etaOrientation) * activatedOrientationError];

    for innerIdx = 1:600
        projectedArgument = kappa - (Q * kappa + B);
        projectedArgument = projectToFeasibleSet(projectedArgument, lowerBound, upperBound, numJoints);
        kappa = kappa + solverStep * (-kappa + projectedArgument);
    end

    qdot = kappa(1:numJoints);
    q = q + qdot * dt;
    qddot = (qdot - qdotPrevious) / dt;
    qdotPrevious = qdot;

    updatedTransform = robot.fkine(q).T;
    updatedPosition = updatedTransform(1:3, 4);
    updatedRotation = updatedTransform(1:3, 1:3);

    alphaPosition = dt / (tauPositionFilter + dt);
    filteredPositionNew = (1 - alphaPosition) * filteredPosition + alphaPosition * updatedPosition;
    estimatedLinearVelocity = (updatedPosition - filteredPositionNew) / tauPositionFilter;

    rotationIncrement = updatedRotation * previousRotationForJacobian';
    estimatedAngularVelocityRaw = logSO3Vector(rotationIncrement) / dt;

    endEffectorVelocityRaw = [estimatedLinearVelocity; estimatedAngularVelocityRaw];

    alphaVelocity = dt / (tauVelocityFilter + dt);
    filteredEndEffectorVelocityNew = (1 - alphaVelocity) * filteredEndEffectorVelocity + alphaVelocity * endEffectorVelocityRaw;
    estimatedEndEffectorAcceleration = (endEffectorVelocityRaw - filteredEndEffectorVelocityNew) / tauVelocityFilter;

    thetaDot = qdot;
    thetaDDot = qddot;
    thetaDotRegularizedInverse = thetaDot' / (thetaDot' * thetaDot + pinvRegularization);

    JEstimatedDot = (estimatedEndEffectorAcceleration - JEstimated * thetaDDot + ...
                     jacobianAdaptationGain * (filteredEndEffectorVelocityNew - JEstimated * thetaDot)) ...
                     * thetaDotRegularizedInverse;

    JEstimated = JEstimated + JEstimatedDot * dt;
    Jp = JEstimated(1:3, :);
    Jo = JEstimated(4:6, :);

    JTrueForPlot = robot.jacob0(q);

    filteredPosition = filteredPositionNew;
    filteredEndEffectorVelocity = filteredEndEffectorVelocityNew;
    previousRotationForJacobian = updatedRotation;

    positionErrorLog(:, idx) = positionError;
    orientationErrorLog(:, idx) = orientationError;
    actualPositionLog(:, idx) = currentPosition;
    desiredPositionLog(:, idx) = desiredPositionCurrent;
    qLog(:, idx) = q;
    qdotLog(:, idx) = qdot;
    qddotLog(:, idx) = qddot;
end

positionErrorNorm = sqrt(sum(positionErrorLog.^2, 1));
orientationErrorNorm = sqrt(sum(orientationErrorLog.^2, 1));

fprintf('\n==================== Proposed Tracking Summary ====================\n');
fprintf('Position RMS error  = %.6f m\n', sqrt(mean(positionErrorNorm.^2)));
fprintf('Position Max error  = %.6f m\n', max(positionErrorNorm));
fprintf('Position Min error  = %.6f m\n', min(positionErrorNorm));
fprintf('Orientation RMS err = %.6f rad\n', sqrt(mean(orientationErrorNorm.^2)));
fprintf('Orientation Max err = %.6f rad\n', max(orientationErrorNorm));
fprintf('Max |qdot|          = %.6f rad/s\n', max(abs(qdotLog(:))));
fprintf('====================================================================\n');


figure;
subplot(2, 1, 1);
lineStyles3 = {'-.', ':', '--'};
for dimIdx = 1:3
    plot(t, positionErrorLog(dimIdx, :), 'LineStyle', lineStyles3{dimIdx}, 'LineWidth', 2.5);
    hold on;
end
ylabel('$\mathbf{e_p}$[m]', 'Interpreter', 'latex');
legend({'$\mathbf{e_p}_x$', '$\mathbf{e_p}_y$', '$\mathbf{e_p}_z$'}, ...
       'Interpreter', 'latex', 'FontSize', 14, 'Orientation', 'horizontal');
formatCurrentAxis();

subplot(2, 1, 2);
for dimIdx = 1:3
    plot(t, orientationErrorLog(dimIdx, :), 'LineStyle', lineStyles3{dimIdx}, 'LineWidth', 2.5);
    hold on;
end
xlabel('t [s]');
ylabel('$\mathbf{e_o}$[rad]', 'Interpreter', 'latex');
legend({'$\mathbf{e_o}_x$', '$\mathbf{e_o}_y$', '$\mathbf{e_o}_z$'}, ...
       'Interpreter', 'latex', 'FontSize', 14, 'Orientation', 'horizontal');
formatCurrentAxis();


figure;
lineStylesEuler = {':', '-.', '--'};
plot(t, desiredEulerLog(1:3, :), 'LineWidth', 2.5);
hold on;
for dimIdx = 1:3
    plot(t, actualEulerLog(dimIdx, :), 'LineStyle', lineStylesEuler{dimIdx}, 'LineWidth', 2.5);
    hold on;
end
xlabel('t [s]');
ylabel('Euler angle [rad]');
legend('$\alpha_d$', '$\beta_d$', '$\gamma_d$', '$\alpha_a$', '$\beta_a$', '$\gamma_a$', ...
       'Interpreter', 'latex', 'Location', 'best', 'NumColumns', 2);
formatCurrentAxis();

figure;
plot(t, desiredEulerLog(1:3, :) - actualEulerLog(1:3, :), 'LineWidth', 2.5);
grid on;
xlabel('t [s]');
ylabel('Euler-angle error [rad]');
legend('$e_{\alpha}$', '$e_{\beta}$', '$e_{\gamma}$', 'Interpreter', 'latex');
formatCurrentAxis();


figure;
hDesiredPath = plot3(desiredPositionLog(1, :), desiredPositionLog(2, :), desiredPositionLog(3, :), ...
                     'r:', 'LineWidth', 2.5);
hold on;
hActualPath = plot3(actualPositionLog(1, :), actualPositionLog(2, :), actualPositionLog(3, :), ...
                    'Color', [0.3, 0.3, 0.3], 'LineWidth', 2.5);

for idx = 1:150:numSteps
    RDesiredCurrent = rotationDesired{idx};
    RActualCurrent = actualRotationLog{idx};

    xDesiredAxis = RDesiredCurrent(:, 1);
    yDesiredAxis = RDesiredCurrent(:, 2);
    zDesiredAxis = RDesiredCurrent(:, 3);

    xActualAxis = RActualCurrent(:, 1);
    yActualAxis = RActualCurrent(:, 2);
    zActualAxis = RActualCurrent(:, 3);

    quiver3(desiredPositionLog(1, idx), desiredPositionLog(2, idx), desiredPositionLog(3, idx), ...
            xDesiredAxis(1), xDesiredAxis(2), xDesiredAxis(3), 0.01, ...
            'LineWidth', 1, 'AutoScale', 'off');
    quiver3(desiredPositionLog(1, idx), desiredPositionLog(2, idx), desiredPositionLog(3, idx), ...
            yDesiredAxis(1), yDesiredAxis(2), yDesiredAxis(3), 0.01, ...
            'g', 'LineWidth', 1, 'AutoScale', 'off');
    quiver3(desiredPositionLog(1, idx), desiredPositionLog(2, idx), desiredPositionLog(3, idx), ...
            zDesiredAxis(1), zDesiredAxis(2), zDesiredAxis(3), 0.01, ...
            'b', 'LineWidth', 1, 'AutoScale', 'off');

    quiver3(actualPositionLog(1, idx), actualPositionLog(2, idx), actualPositionLog(3, idx), ...
            xActualAxis(1), xActualAxis(2), xActualAxis(3), 0.01, ...
            '--', 'LineWidth', 1, 'AutoScale', 'off');
    quiver3(actualPositionLog(1, idx), actualPositionLog(2, idx), actualPositionLog(3, idx), ...
            yActualAxis(1), yActualAxis(2), yActualAxis(3), 0.01, ...
            ':', 'LineWidth', 1, 'AutoScale', 'off');
    quiver3(actualPositionLog(1, idx), actualPositionLog(2, idx), actualPositionLog(3, idx), ...
            zActualAxis(1), zActualAxis(2), zActualAxis(3), 0.01, ...
            '-.', 'LineWidth', 1, 'AutoScale', 'off');
end

grid on;
box off;
xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
legend([hDesiredPath, hActualPath], {'$\mathbf{p}_d$', '$\mathbf{p}$'}, ...
       'Interpreter', 'latex', 'Location', 'north', 'NumColumns', 2);
formatCurrentAxis();
view(-45, 25);


figure;
subplot(2, 1, 1);
lineStyles6 = {'-', '-', '--', '--', '-.', '-.'};
for jointIdx = 1:numJoints
    plot(t, qLog(jointIdx, :), 'LineStyle', lineStyles6{jointIdx}, 'LineWidth', 2.5);
    hold on;
end
ylabel('$\theta$[rad]', 'Interpreter', 'latex');
formatCurrentAxis();

subplot(2, 1, 2);
for jointIdx = 1:numJoints
    plot(t, qdotLog(jointIdx, :), 'LineStyle', lineStyles6{jointIdx}, 'LineWidth', 2.5);
    hold on;
end
xlabel('t [s]');
ylabel('$\dot{\theta}$[rad/s]', 'Interpreter', 'latex', 'FontSize', 14);
formatCurrentAxis();

