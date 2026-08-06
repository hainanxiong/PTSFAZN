# Reproducible MATLAB Code for Pose-Tracking Experiments

## Overview

This repository provides the MATLAB implementation used to reproduce the simulation, comparison, and robustness-evaluation results reported in the accompanying manuscript. The code includes the main pose-tracking simulation, auxiliary functions, comparisons with zeroing-neurodynamics-based methods, comparisons with existing control schemes, and robustness tests.

The repository is organized so that the main simulation and each group of comparison experiments can be executed independently.

## Repository Structure

```text
.
├── Comparison with ZN/          # Comparisons with ZN-based baseline methods
├── Results/                     # Saved simulation results and/or generated figures
├── Robustness_Evaluation/       # Robustness tests under model/kinematic variations
├── Scheme_Comparison/           # Comparisons with existing control schemes
├── formatCurrentAxis.m          # Common axis-formatting utility for figures
├── fuzzyErrorToEta.m            # Fuzzy mapping from tracking error to parameter eta
├── generalizedBellMf.m          # Generalized bell-shaped membership function
├── logSO3Vector.m               # SO(3) logarithmic-map vector computation
├── mainSimulation.m             # Main simulation entry point
├── projectToFeasibleSet.m       # Projection onto the prescribed feasible set
├── vee.m                        # Vee operator for skew-symmetric matrices
└── zdActivation.m               # Zeroing-dynamics activation function
```

## Requirements

- MATLAB.
- Peter Corke's Robotics Toolbox for MATLAB, including the PUMA 560 model (`mdl_puma560`) and the required kinematic routines.
- MATLAB rotation-conversion functions such as `axang2rotm`, `rotm2axang`, and `eul2rotm` for the scripts that use axis-angle or Euler-angle representations.

Before running the simulations, make sure that the required toolboxes are installed and added to the MATLAB path.

## Quick Start

1. Clone or download this repository.
2. Open MATLAB and set the repository root as the current folder.
3. Make sure Peter Corke's Robotics Toolbox is available on the MATLAB path.
4. Run the main simulation:

```matlab
mainSimulation
```

The main script calls the auxiliary functions in the repository root as required.

## Comparison Experiments

### 1. Comparison with ZN-Based Methods

The `Comparison with ZN/` directory contains the simulations used to compare the proposed approach with alternative zeroing-neurodynamics-based solvers, including the LZN and PTCZN baselines used in the manuscript.

Each comparison script explicitly defines its trajectory, controller parameters, numerical settings, constraints, and activation function to facilitate independent verification.

### 2. Comparison with Existing Control Schemes

The `Scheme_Comparison/` directory contains the implementations used for scheme-level comparisons with existing neural-dynamics-based pose-tracking approaches discussed in the manuscript.

The competing methods are implemented as independent MATLAB scripts so that their assumptions, controller parameters, and numerical settings can be inspected directly.

### 3. Robustness Evaluation

The `Robustness_Evaluation/` directory contains the robustness experiments, including simulations with time-varying kinematic/D-H-parameter perturbations where applicable.

All perturbation parameters are defined explicitly in the corresponding MATLAB scripts.

## Results

The `Results/` directory contains the simulation outputs used for result visualization and comparison. The plotting routines in the main and comparison scripts reproduce the corresponding tracking trajectories, position/orientation errors, and other evaluation quantities.

## Reproducibility and Fair Comparison

To make the comparisons transparent and reproducible:

- Common task definitions and simulation settings are stated explicitly in the relevant scripts.
- Method-specific design parameters are kept in the corresponding implementation rather than hidden in external configuration files.
- Exact/ground-truth kinematic quantities used only for simulation or evaluation are distinguished from quantities supplied to the controller whenever applicable.
- Position and orientation tracking errors are recorded separately for quantitative comparison.
- The implementations of the competing methods are provided together with the proposed method so that the comparison procedure can be inspected directly.

For exact parameter values, please refer to the parameter section at the beginning of each MATLAB script.

## Main Utility Functions

| File | Purpose |
| --- | --- |
| `formatCurrentAxis.m` | Applies the common plotting/axis style used in the simulations. |
| `fuzzyErrorToEta.m` | Maps the tracking-error information to the adaptive parameter `eta`. |
| `generalizedBellMf.m` | Evaluates the generalized bell-shaped membership function used by the fuzzy mechanism. |
| `logSO3Vector.m` | Computes a vector representation of the logarithmic map on SO(3). |
| `projectToFeasibleSet.m` | Projects the optimization/neural state onto the prescribed feasible set. |
| `vee.m` | Implements the vee operator for a skew-symmetric matrix. |
| `zdActivation.m` | Implements the zeroing-dynamics activation function used by the controller. |

## Notes for Reproduction

- Run each experiment from the repository root or preserve the original relative directory structure.
- Do not change controller gains, initial conditions, constraints, or sampling intervals when reproducing the reported results unless performing a separate sensitivity study.
- Some comparison methods have method-specific parameters required by their original formulations; these are documented directly in their scripts.

## References

The comparison implementations correspond to the methods cited in the accompanying manuscript. The relevant source method is identified in the header of each comparison script. Please refer to the manuscript for the complete bibliographic information.

## Questions

For questions regarding reproduction of the simulations, please use the repository issue tracker.
