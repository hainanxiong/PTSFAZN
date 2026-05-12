function y = generalizedBellMf(x, a, b, c)
    y = 1 ./ (1 + abs((x - c) ./ a).^(2 * b));
end