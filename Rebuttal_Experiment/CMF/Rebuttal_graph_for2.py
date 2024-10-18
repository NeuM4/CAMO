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
        
        'GP_UCB': ['#4169E1', "^", "BOCA", 'solid'],
        'GP_cfKG': ['#4169E1', "X", "cfKG", 'solid'],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAMO-BOCA", 'dashed'], # red
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", 'dashed'],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-BOCA", 'dashed'], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", 'dashed'],
        }

max_dic = {'non_linear_sin':0, 'forrester': 50,'Branin': 55,'Currin': 14,'Park': 2.2,'colvile':609.26,
           'himmelblau':303.5,'bohachevsky':72.15,'borehole':244}
add_dic = {'colvile': 125, 'himmelblau': 1,'borehole':1}
lim_x = {'borehole':[48,150],'colvile': [48, 150], 'himmelblau': [48, 150]}
lim_y = {'borehole':[0, 110],'colvile': [0, 425], 'himmelblau': [0, 150]}
seed_dic = {'borehole':[1,2,7,12,17],'colvile': [0,4,5,8,9], 'himmelblau': [0,1,2,3,8]}

cmf_methods_name_list = ['GP_UCB', 'GP_cfKG', 
                         
                         'CMF_CAR_UCB',
                         'CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB', 
                         'CMF_CAR_dkl_cfKG',
                         'fabolas','smac',
                         ]

data_list = ['borehole', 'colvile', 'himmelblau']
cost_name = 'pow_10'
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
Exp_marker = 'Norm_res'

for kk in range(3):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            if methods_name == 'fabolas':
                SR = max_dic[data_name]-data['incumbents'].to_numpy()
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
    
        ll = axs[kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=17)
        axs[kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs[kk].set_xlabel("Cost", fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    axs[kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()

lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.23), fancybox=True, mode='normal', ncol=4, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
# plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graph_show') + '/' + 'CMF_rebuttal3_' + cost_name +'_SR_together.pdf', bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graph_show') + '/' + 'CMF_rebuttal3_' + cost_name +'_SR_together.eps', bbox_inches='tight')

plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'ICLR_res') + '/' + 'CMF_rebuttal3_' + cost_name +'_SR_together.pdf', bbox_inches='tight')