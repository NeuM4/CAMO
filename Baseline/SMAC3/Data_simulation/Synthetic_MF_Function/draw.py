import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import matplotlib.pyplot as plt
from Data_simulation.Synthetic_MF_Function import *


# 实例化非线性sin函数类
# non_linear_sin_instance = non_linear_sin(cost_type=None, total_fidelity_num=2)
# instance = forrester(cost_type='pow_10', total_fidelity_num=2)
# instance = maolin1(cost_type='pow_10', total_fidelity_num=2)
# instance = tl2(cost_type='pow_10', total_fidelity_num=2)
instance = booth(cost_type='pow_10', total_fidelity_num=2)

# 初始化数据
xtr, ytr = instance.Initiate_data(index=[1000, 1000], seed=0)

# 绘制函数曲线
plt.figure(figsize=(10, 6))
plt.scatter(xtr[0], ytr[0], label='Low Fidelity', color='blue')
plt.scatter(xtr[1], ytr[1], label='High Fidelity', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('booth Function')
plt.legend()
plt.grid(True)
plt.savefig('booth.png')
