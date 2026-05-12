function z = projectToFeasibleSet(x, lowerBound, upperBound, numJoints)
    z = x;
    z(1:numJoints) = min(max(x(1:numJoints), lowerBound), upperBound);
end