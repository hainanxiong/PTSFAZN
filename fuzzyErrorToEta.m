function etaValue = fuzzyErrorToEta(errorValue)
    outputScale = 0.01;

    etaDomain = linspace(0, 5, 1200);
    mfZero = 1 ./ (1 + exp((etaDomain - 0.2) / 0.04));
    mfPositiveSmall = generalizedBellMf(etaDomain, 0.4, 2.8, 1.65);
    mfPositiveMedium = generalizedBellMf(etaDomain, 0.4, 2.8, 3.1);
    mfPositiveBig = 1 ./ (1 + exp(-(etaDomain - 4) / 0.2));

    wZero = exp(-((errorValue - 0.000).^2) / (2 * 0.05^2));
    wPositiveSmall = exp(-((errorValue - 0.18).^2) / (2 * 0.04^2));
    wPositiveMedium = exp(-((errorValue - 0.32).^2) / (2 * 0.04^2));
    wPositiveBig = 1 ./ (1 + exp(-(errorValue - 0.45) / 0.06));

    cutZero = min(wZero, mfZero);
    cutPositiveSmall = min(wPositiveSmall, mfPositiveSmall);
    cutPositiveMedium = min(wPositiveMedium, mfPositiveMedium);
    cutPositiveBig = min(wPositiveBig, mfPositiveBig);

    aggregatedMf = max(max(cutZero, cutPositiveSmall), max(cutPositiveMedium, cutPositiveBig));

    if sum(aggregatedMf) == 0
        etaValue = 0;
    else
        etaValue = outputScale * sum(aggregatedMf .* etaDomain) / sum(aggregatedMf);
    end
end