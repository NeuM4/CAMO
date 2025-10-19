import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data


# UCB * EI s cfkg o
Dic = {
    'fabolas': ['#808000', "*", "Fabolas", 'solid'],  # 深红 + 五边形
    'smac': ['#2E8B57', "H", "SMAC3", 'dashdot'],     # 深绿色 + 六边形
    
    'GP_UCB': ['#1E90FF', "s", "BOCA", 'solid'],       # 深蓝 + 方形
    'GP_cfKG': ['#1E90FF', "o", "cfKG", 'solid'],      # 深蓝 + 圆形
    
    'CMF_CAR_UCB': ['#FF4500', "D", "CAMO-BOCA", 'dashed'],   # 橙红 + 菱形
    'CMF_CAR_cfKG': ['#FF4500', "v", "CAMO-cfKG", 'dashed'],  # 橙红 + 下三角形
    
}



max_dic = {'Branin': 55,'Currin': 13.798,'Park': 2.2, 'VibratePlate': 250, 'HeatedBlock': 2,'bohachevsky': 72.15,'borehole':0}
add_dict = {'Branin': 3,'Currin': 0.02,'Park': 0.3, 'VibratePlate': 0, 'HeatedBlock': 0,'bohachevsky': 4,'borehole':0}
cost_lim_y = {'Branin': [2,12], 'Currin': [0, 3], 'Park': [0.2, 1.4],'bohachevsky': [0, 32], 'borehole': [0, 0.5]}
cost_lim_x = {'Branin': [13, 300], 'Currin': [13, 150], 'Park': [13, 300],'bohachevsky': [13, 150], 'borehole': [13, 300]}


##Currin
data_name = 'Currin'
# data_name = 'bohachevsky'
seed_dic ={'pow_10':[i for i in range(30)],'linear':[i for i in range(30)],'log':[i for i in range(30)]}

methods_name_list = [ 
                     'GP_UCB', 
                     'GP_cfKG',
                     'fabolas',
                     'smac',
                     'CMF_CAR_UCB',
                    ]

cost_list = ['log', 'linear', 'pow_10']
cost_label_dic = {'log': 'Log', 'linear': 'Linear', 'pow_10': 'Power 10'}
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

Exp_marker = "Norm_res"

for kk in range(3):
    cost_name = cost_list[kk]
    for methods_name in methods_name_list:
        print(methods_name)
        cost_collection = []
        # SR_collection = []
        inter_collection = []  
        for seed in seed_dic[cost_name]:
            path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            
            if methods_name == 'fabolas':
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
        var = np.std(SR_new, axis=0)
        if  cost_name == 'log':
            if methods_name in ['fabolas']:
                makervery_index = 120
            else:
                makervery_index = 360
        elif  cost_name == 'linear':
            makervery_index = 30
        else:
            makervery_index = 30
        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=makervery_index,linewidth=2.2)
        
    axs[kk].set_xlabel("Cost: " + cost_label_dic[cost_name], fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    # axs[kk].set_yscale('log')
    axs[kk].set_xlim(cost_lim_x[data_name][0], cost_lim_x[data_name][1])
    axs[kk].set_ylim(cost_lim_y[data_name][0], cost_lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()

# label = [Dic[i][-1] for i in methods_name_list]
# label = label + [Dic[i+'_con'][-1] for i in cmf_methods_name_list]

# 共享图例
lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, mode='normal', ncol=5, markerscale = 1.3, fontsize=25)

# change the line width for the legend
# for line in leg.get_lines():
#     line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Graphs') + '/' +'CMF_' + data_name + '_cost.png', bbox_inches='tight')
