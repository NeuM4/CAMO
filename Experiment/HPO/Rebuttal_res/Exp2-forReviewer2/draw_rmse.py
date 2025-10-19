import pandas as pd
import matplotlib.pyplot as plt
import os

# 初始化存储数据的列表
levels = []
cam_rmse = []
gp_rmse = []

# 读取所有level文件
for i in range(11):  # 从level0.0到level1.0共11个文件
    file_name = f"res_level{i/10:.1f}.csv"
    if not os.path.exists(file_name):
        continue
    
    # 读取CSV文件
    df = pd.read_csv(file_name)
    
    # 确保文件中有足够的数据行
    if len(df) >= 2:
        levels.append(i/10)
        cam_rmse.append(df['rmse'].iloc[0])  # 第一行是CAMO的数据
        gp_rmse.append(df['rmse'].iloc[1])   # 第二行是GP-SE的数据

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(levels, cam_rmse, 'r-s', label='CAMO')
plt.plot(levels, gp_rmse,'b--o', label='GP-SExSE')

# 添加图表标题和标签
plt.title('RMSE Comparison across Different fidelity Levels (Data: bohachevsky)')
plt.xlabel('fidelity Level')
plt.ylabel('RMSE')
plt.xticks([i/10 for i in range(11)])  # 设置x轴刻度为0.0到1.0，步长0.1
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 显示图表
plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300)