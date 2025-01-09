# CAMO: CONVERGENCE-AWARE MULTI-FIDELITY BAYESIAN OPTIMIZATION
💫We propose CAMO (Convergence-Aware Multi-fidelity Optimization), a novel MFBO framework based on Fidelity Differential Equations (FiDEs). CAMO explicitly captures the convergence behavior of the objective function through rigorous theoretical analysis, enabling more efficient optimization across fidelity levels. Our approach allows for automatic adaptation to varying fidelity levels without extensive manual intervention, making it particularly suitable for practical MFBO scenarios where the highest and lowest fidelity levels may be unknown or difficult to specify.
# Algorithm performance
The performance of CAMO in terms of simple regret on the Borehole, Colville, and Himmelblau datasets is as follows. It can be observed that our method converges to the optimal solution the fastest across all three datasets, highlighting the importance of convergence awareness. More experimental results can be found in the paper.
<p align="center">
  <img src="Experiment\CMF\Graphs\high_dim_show.png" width="750">
</p>

# 🗣️Project Structure
```
project_root/
├── README.md
├── requirements.txt
├── Acq (acquisition function)
├── assert (some figure and data)
├── Baseline (fabolas and smac) 
├── Data_simulation (cost function and synthetic mf function)
├── Experiment
├── FidelityFusion_models(surrogate model)
├── GaussionProcess
```

# 📝Run
## ⛏️Installation
We recommend using a virtual environment.
```
git clone https://github.com/IceLab-X/CAMO.git
cd CAMO 
pip install .
```
## Perform CAMO in synthetic MF function
We support simulation data of 10+, Cost_function can be selected from pow_10, linear, or log
```
cd Experiment\CMF
python CMF_norm.py -- "data_name" --cost_type "pow_10"
```

**Visualization of experimental results**

We provide graphic code
```
python Graph_for_cmf_norm.py
```

## Perform CAMO in real world simulation
In the real data experiments, we used Python to call MATLAB scripts. Therefore, when running the real data experiments, please ensure that MATLAB and the MATLAB Engine API for Python are installed on your computer.
We provide two usage examples in real-world scenarios: HeatedBlock and VibratePlate
```
python CMF_real.py -- "data_name" --cost_type "pow_10"
```

## 🖼️Baseline
We provide two Bayesian optimization baselines, Fabolas and SMAC, for continuous problems. For the baseline DNN for discrete problems, we built it based on the code repository from [Shibo](https://github.com/shib0li/DNN-MFBO)

**Perform fabolas**

```
cd Baseline/Fabolas
python fabolas_demo.py -- "data_name" --cost_type "pow_10"
```
**Perform smac3**

For SMAC, we built it based on SMAC3 from AutoML. The related environment dependencies can be found in the original [SMAC](https://github.com/automl/SMAC3) repository and its documentation.
```
cd Baseline/smac_camo
python smac_demo.py -- "data_name"
```


# 💐Contributing to CAMO 
- **Reporting bugs.** To report a bug, simply open an issue in the GitHub [Issues](https://github.com/IceLab-X/CAMO/issues).
- **Suggesting enhancements.** To submit an enhancement suggestion, including completely new features or minor improvements on existing features, please open an issue in the GitHub [Issues](https://github.com/IceLab-X/CAMO/issues).
- **Pull requests.** If you made improvements to FidelityFusion, fixed a bug, or had a new example, feel free to send us a pull-request.
- **Asking questions.** To get help on how to use FidelityFusion or its functionalities, you can open a discussion in the GitHub.
# 👷‍♂️The Team

# 🤗Citation
💥Please cite our paper if you find it helpful :) 