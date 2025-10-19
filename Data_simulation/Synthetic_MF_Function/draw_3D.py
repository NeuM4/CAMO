import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from Data_simulation.Synthetic_MF_Function import *


instance = Branin(cost_type='pow_10')

xtr, ytr = instance.Initiate_data(index=[1000, 1000, 1000], seed=0)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xtr[0][:, 0], xtr[0][:, 1], ytr[0], label='Low Fidelity', color='blue')
ax.scatter(xtr[1][:, 0], xtr[1][:, 1], ytr[1], label='Middle Fidelity', color='yellow')
ax.scatter(xtr[2][:, 0], xtr[2][:, 1], ytr[2], label='High Fidelity', color='red')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Brainin Function')

plt.legend()

# 保存图形
# plt.savefig('Brainin_3D.png')
