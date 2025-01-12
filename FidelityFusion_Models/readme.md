# FidelityFusion_Models Modules
In this section, we provide the code for the models used in our library.

## Supported models
### 1. AR (AutoRegression)
The method can be found in the following paper
[Predicting the Output from a Complex Computer Code When Fast Approximations Are Available](https://www.jstor.org/stable/2673557)
### 2. ResGP(Residual Gaussian Process)
The method can be found in the following paper
[Residual Gaussian process: A tractable nonparametric Bayesian emulator for multi-fidelity simulations](https://www.sciencedirect.com/science/article/abs/pii/S0307904X21001724)

### 3. CAR (Continue Autoregression)
[ContinuAR: Continuous Autoregression For Infinite-Fidelity Fusion](https://openreview.net/pdf?id=wpfsnu5syT)

## 🗣️Folder Explanation
```
FidelityFusion_Models/
├── readme.md
├── CMF_CAR_dkl.py (Continue DKL surrogate model for CAMO)
├── CMF_CAR.py (Continue Analytical solution implementation for CAMO)
├── DMF_CAR_dkl.py (Discrete DKL surrogate model for CAMO)
├── DMF_CAR.py (Discrete Analytical solution implementation for CAMO)
├── MF_data.py (Class for Multi-fidelity data management)
├── GP_dkl.py (Deep kernel GP)
├── GP_DMF.py (Discrete fidelity Implementation of GP)
```
