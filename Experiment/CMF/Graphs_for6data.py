import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerLine2D
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data

def draw_plots(axs, data_name, cmf_methods_name_list, exp_marker):
    label_name = []
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []
        
        if methods_name in ['smac']:
            for seed in seed_dic[data_name]:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',exp_marker,
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                data = pd.DataFrame(pd.read_csv(path))
                cost = data['cost'].to_numpy()
                SR = data['SR'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
                cost_collection.append(cost)
                inter_collection.append(inter)
        else:
            for seed in seed_dic[data_name]:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',exp_marker,
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                data = pd.DataFrame(pd.read_csv(path))
                cost = data['cost'].to_numpy()
                if methods_name in ['fabolas', 'smac']:
                    SR = max_dic[data_name] - data['incumbents'].to_numpy()
                else:
                    SR = data['SR'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
                cost_collection.append(cost)
                inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)

        new_method_name = methods_name
        label_name.append(new_method_name)
        
        axs.plot(cost_x, mean + add_dic[data_name], ls=Dic[new_method_name][-1], color=Dic[new_method_name][0],
                    label=Dic[new_method_name][2],
                    marker=Dic[new_method_name][1], markersize=12, markevery=30,linewidth=2)


    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs.tick_params(axis='both', labelsize=20)
    axs.grid()

    return label_name

# UCB * EI s cfkg o
Dic = {
    'fabolas': ['#808000', "*", "Fabolas", 'solid'],  # 深红 + 五边形
    'smac': ['#2E8B57', "H", "SMAC3", 'dashdot'],     # 深绿色 + 六边形
    
    'GP_UCB': ['#1E90FF', "s", "BOCA", 'solid'],       # 深蓝 + 方形
    'GP_cfKG': ['#1E90FF', "o", "cfKG", 'solid'],      # 深蓝 + 圆形
    
    'CMF_CAR_UCB': ['#FF4500', "D", "CAMO-BOCA", 'dashed'],   # 橙红 + 菱形
    'CMF_CAR_cfKG': ['#FF4500', "v", "CAMO-cfKG", 'dashed'],  # 橙红 + 下三角形
}


data_list = ['Branin', 'Currin', 'Park', 'non_linear_sin', 'Forrester','bohachevsky']
cost_name = 'pow_10'

max_dic = {'forrester': 48.4495, 'non_linear_sin':0.03338,'Branin': 54.75,'Currin': 13.798,'Park': 2.174,'himmelblau':303.5,'bohachevsky': 72.15}
add_dic = {'forrester': 0.8 , 'non_linear_sin': 0,'Branin': 0.9,'Currin': 0.01,'Park': 0.1, 'himmelblau': 1,'bohachevsky': 4}
lim_x = {'forrester': [48, 300], 'non_linear_sin': [48, 300], 'Branin':[48,300],'Currin':[48,300],'Park':[48,300], 'himmelblau':[48, 300],'bohachevsky':[48,300]}
lim_y = {'forrester': [0, 25], 'non_linear_sin': [0,0.04], 'Branin':[0,8], 'Currin':[0,2],'Park':[0,1],'himmelblau':[0, 150],'bohachevsky':[0,22]}

seed_dic ={
           'Currin':[i for i in range(30)],
           'Branin':[i for i in range(30)],
           'Park':[i for i in range(30)],
           'non_linear_sin':[i for i in range(30)],
           'forrester':[i for i in range(30)],
           'bohachevsky':[i for i in range(30)]}

cmf_methods_name_list = [  
                        'GP_UCB','GP_cfKG',
                        'CMF_CAR_UCB','CMF_CAR_cfKG',
                        'fabolas',
                        'smac'
                         ]
Exp_marker = 'Norm_res'


fig = plt.figure(figsize=(30, 16))

# 创建图形
gs = gridspec.GridSpec(2, 6) # 创立2 * 6 网格
gs.update(wspace=0.9)

# 对第一行进行绘�?
ax1 = plt.subplot(gs[0,  :2]) # gs(哪一行，绘制网格列的范围)
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4:6])

# 对第二行进行绘制
ax4 = plt.subplot(gs[1, :2])
ax5 = plt.subplot(gs[1, 2:4])
ax6 = plt.subplot(gs[1, 4:6])

# 开始画�?
draw_plots(ax1, 'Branin', cmf_methods_name_list, Exp_marker)
ax1.text(0.5, 1.02, 'Branin', transform=ax1.transAxes, ha='center', fontsize=25)
draw_plots(ax2, 'Currin', cmf_methods_name_list, Exp_marker)
ax2.text(0.5, 1.02, 'Currin', transform=ax2.transAxes, ha='center', fontsize=25)
draw_plots(ax3, 'Park', cmf_methods_name_list, Exp_marker)
ax3.text(0.5, 1.02, 'Park', transform=ax3.transAxes, ha='center', fontsize=25)
draw_plots(ax4, 'non_linear_sin', cmf_methods_name_list, Exp_marker)
ax4.text(0.5, 1.02, 'Non-linear Sin', transform=ax4.transAxes, ha='center', fontsize=25)
draw_plots(ax5, 'forrester', cmf_methods_name_list, Exp_marker)
ax5.text(0.5, 1.02, 'Forrester', transform=ax5.transAxes, ha='center', fontsize=25)
draw_plots(ax6, 'bohachevsky', cmf_methods_name_list, Exp_marker)
ax6.text(0.5, 1.02, 'Bohachevsky', transform=ax6.transAxes, ha='center', fontsize=25)

lines, labels = ax1.get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, mode='normal', ncol=6, markerscale = 1.5, fontsize=28)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Graphs') + '/' + 'CMF_' + cost_name +'_SR_6.png', bbox_inches='tight')