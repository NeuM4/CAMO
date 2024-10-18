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
Dic = { 'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac':['#006400', "*", "SMAC3", 'solid'],
        
        'GP_UCB_con': ['#4169E1', "^", "BOCA", 'solid'],
        'GP_cfKG_con': ['#4169E1', "X", "cfKG", 'solid'],
        
        'CMF_CAR_UCB_con': ['#FF0000', "^", "CAMO-BOCA", 'dashed'], # red
        'CMF_CAR_cfKG_con': ['#FF0000', "X", "CAMO-cfKG", 'dashed'],
        'CMF_CAR_dkl_UCB_con': ['#FF5E00', "^", "CAMO-DKL-BOCA", 'dashed'], # orange
        'CMF_CAR_dkl_cfKG_con': ['#FF5E00', "X", "CAMO-DKL-cfKG", 'dashed'],
        }


add_dic = {'VibratePlate2': 0, 'HeatedBlock': 1}
max_dic = {'VibratePlate': 250, 'HeatedBlock': 2}
lim_x = {'VibratePlate2': [80,350], 'HeatedBlock': [20, 200]}
lim_y = {'VibratePlate2': [28, 41.8], 'HeatedBlock': [0,1.8]}
seed_dic = {'VibratePlate2': [0,1,2,4,5], 'HeatedBlock': [1,2,3,6]}

cmf_methods_name_list = ['GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB','CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']
baseline_list = ['fabolas','smac']
baseline_seed_dic = {'VibratePlate':[0,1,2,3,4,5,7,8,9],'HeatedBlock':[0,1,2,3,4,5,7,8,9]}

data_list = ['VibratePlate2', 'HeatedBlock']
cost_name = 'pow_10'
fig, axs = plt.subplots(1, 2, figsize=(20, 6))


lim_x = {'VibratePlate2': [80, 350], 'HeatedBlock': [22, 200]}
for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost_tem = data['operation_time'].to_numpy()
            cost = np.cumsum(cost_tem)
            SR = data['SR'].to_numpy()
            # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        
        methods_name = methods_name + '_con'

        ll = axs[kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                label=Dic[methods_name][2],
                marker=Dic[methods_name][1], markersize=12,markevery=14)
        axs[kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
    
    for methods_name in baseline_list:
        cost_collection = []
        inter_collection = []
        

        if data_name == 'VibratePlate2':
            data_name = 'VibratePlate'
        for seed in [0,1,2,3,4,]:
            

            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost_tem = data['operation_time'].to_numpy()
            cost = np.cumsum(cost_tem)
            if methods_name == 'fabolas':
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy()
            # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)
        
        if data_name == 'VibratePlate':
            data_name = 'VibratePlate2'

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        
        methods_name = methods_name
        ll = axs[kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=14)
        axs[kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs[kk].set_xlabel("Wall clock time (s)", fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    axs[kk].set_xscale('log')
    # axs[kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()


lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, mode='normal', ncol=4, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graph_show') + '/' + 'CMF_real_' + cost_name +'_Rebuttal2.pdf', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graph_show') + '/' + 'CMF_real_' + cost_name +'_Rebuttal2.eps', bbox_inches='tight')