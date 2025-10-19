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

max_dic = {'non_linear_sin':0, 'forrester': 50,'Branin': 55,'Currin': 14,'Park': 2.2,'colvile':609.26,
           'himmelblau':303.5,'borehole':244,'bohachevsky':72.15}
add_dic = {'colvile': 125, 'himmelblau': 1,'borehole':4}
lim_x = {'borehole':[48,300],'colvile': [48, 300], 'himmelblau': [48, 300]}
lim_y = {'borehole':[0, 100],'colvile': [0, 400], 'himmelblau': [0, 120]}

seed_dic = {'himmelblau':[i for i in range(1,30)],
            'borehole':[i for i in range(1,30)],
           'colvile':[i for i in range(1,30)]}


cmf_methods_name_list = ['GP_UCB', 'GP_cfKG', 
                         
                         'CMF_CAR_UCB',
                         'CMF_CAR_cfKG',
                         'fabolas','smac',
                         ]

data_list = ['borehole', 'colvile', 'himmelblau']
cost_name = 'pow_10'
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
Exp_marker = 'Norm_res_60'

for kk in range(3):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results',Exp_marker,
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
                    marker=Dic[methods_name][1], markersize=12,markevery=25,linewidth=2)
        
    axs[kk].set_xlabel("Cost", fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    axs[kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].text(0.5, 1.02, data_name, transform=axs[kk].transAxes, ha='center', fontsize=25)
    axs[kk].grid()


lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, mode='normal', ncol=6, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()

plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF','Graphs') + '/' + 'CMF_hard3_' + cost_name +'_SR_together.png', bbox_inches='tight')