function F = zdActivation(x, rho)
    normX = norm(x, 2);
    normalizedDirection = x ./ (normX + eps);
    sigHigh = normX^(1 + rho) * normalizedDirection;
    sigLow = normX^(1 - rho) * normalizedDirection;
    F = sigHigh + sigLow;
end