function v = logSO3Vector(R)
    c = (trace(R) - 1) / 2;
    c = max(min(c, 1), -1);
    theta = acos(c);

    if theta < 1e-9
        v = [0; 0; 0];
    else
        v = theta / (2 * sin(theta)) * vee(R - R');
    end
end