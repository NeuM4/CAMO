import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# UCB * EI s cfkg o
Dic = { 'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac':['#006400', "*", "SMAC3", 'solid'],
        
        'GP_UCB': ['#4169E1', "^", "MF-GP-UCB", 'solid'],
        'GP_cfKG': ['#4169E1', "X", "MF-GP-cfKG", 'solid'],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAMO-UCB", 'dashed'], # red
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", 'dashed'],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-UCB", 'dashed'], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", 'dashed'],
        }


add_dic = {'VibratePlate': 0, 'HeatedBlock': 0}
lim_x = {'VibratePlate': [70,360], 'HeatedBlock': [25,200]}
lim_y = {'VibratePlate': [28, 41.8], 'HeatedBlock': [0,1.44]}
seed_dic = {'VibratePlate': [0,1,2,3,4,5,7,8,9], 'HeatedBlock': [2,3,5,7,8,9]}
max_dic = {'VibratePlate': 250, 'HeatedBlock': 2}

cmf_methods_name_list = [
                         'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB','CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG',
                         'fabolas',
                         ]

data_list = ['VibratePlate', 'HeatedBlock']
cost_name = 'pow_10'
Exp_marker = 'Norm_res'
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['operation_time'].to_numpy()
            cost = np.cumsum(cost)
            if methods_name in ['fabolas', 'smac']:
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)

        ll = axs[kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=17)
        axs[kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs[kk].set_xlabel("Wall clock time (s)", fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    # axs[kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    # axs[kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[kk].set_xscale('log')
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()

lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, mode='normal', ncol=4, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graph_show') + '/' + 'CMF_real_' + cost_name +'_Walltime.pdf', bbox_inches='tight')
