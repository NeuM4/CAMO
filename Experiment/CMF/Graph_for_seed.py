import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

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

data_list = ['non_linear_sin', 'Forrester', 'Branin', 'Currin', 'Park']
cost_name = 'pow_10'

max_dic = {'Forrester': 50, 'non_linear_sin':0,'Branin': 55,'Currin': 14,'Park': 2.2,
           'himmelblau':303.5,'bohachevsky':72.15,'borehole':244,'colvile':609.26,'bohachevsky': 72.15}
# opt_dic = {'Forrester': 48.4998, 'non_linear_sin':0.133398,'Branin': 54.7544,'Currin': 13.7978,'Park': 2.1736}
add_dic = {'Forrester': 7 , 'non_linear_sin': 0.003,'Branin': 0.85,'Currin': 0.01,'Park': 0,'colvile': 125, 
           'himmelblau': 1,'borehole':10,'bohachevsky': 4}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [48, 135], 'Branin':[48,140],'Currin':[48,140],'Park':[48,140],
         'borehole':[48, 135],'colvile': [48, 135], 'himmelblau': [48, 135],'bohachevsky':[48,140]}
# lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0,0.035], 'Branin':[2,12], 'Currin':[0,1.75],'Park':[0,1.2],
#          'borehole':[0, 125],'colvile': [0, 420], 'himmelblau': [48, 135]}
seed_dic = {'Forrester': [0,1,2,3], 'non_linear_sin': [1,4,5,6,9], 'Branin':[2,5,7,8], 'Currin':[2,3,4,5,6,7,8],'Park':[0,2,4,5,7],
            'borehole':[1,2,7,12,17],'colvile': [0,4,5,8,9], 'himmelblau': [0,1,2,3,8],'bohachevsky':[1,17,19,21,22]}

cmf_methods_name_list = [
                         'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB',
                         'CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB',
                         'CMF_CAR_dkl_cfKG',
                         'fabolas', 'smac',
                         ]

cost_name = 'pow_10'
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    
data_name = 'bohachevsky'

def draw_seed(axs, seed, seed_smac, data_name):
    label_name = []
    for methods_name in cmf_methods_name_list:
        if methods_name == 'smac':
           path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results','Norm_res',
                            data_name, cost_name, methods_name + '_seed_' + str(seed_smac) + '.csv')
        else: 
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy()
        if methods_name in ['fabolas']:
            SR = max_dic[data_name] - data['incumbents'].to_numpy()
        # elif methods_name in ['smac']:
        #     SR = (max_dic[data_name] - data['incumbents'].to_numpy())
        else:
            SR = data['SR'].to_numpy()

        # if methods_name in ['fabolas', 'smac']:
        #     continue
        # else:
        #     SR = np.insert(SR, 0, opt_dic[data_name])
        #     cost_x = np.insert(cost, 0, 50)

        if methods_name in ['fabolas', 'smac']:
            new_method_name = methods_name
            label_name.append(new_method_name)
        else:
            new_method_name = methods_name + '_con'
            label_name.append(new_method_name)

        axs.plot(cost, (SR + add_dic[data_name]), ls=Dic[new_method_name][-1], color=Dic[new_method_name][0],
            label=Dic[new_method_name][2],
            marker=Dic[new_method_name][1], markersize=12)
                
 
    # plt.yscale('log')
    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    # axs.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs.tick_params(axis='both', labelsize=20)
    axs.grid()

draw_seed(axs[0, 0], seed_dic[data_name][0],seed_dic[data_name][0], data_name)
draw_seed(axs[0, 1], seed_dic[data_name][1],seed_dic[data_name][1], data_name)
draw_seed(axs[1, 0], seed_dic[data_name][2],seed_dic[data_name][2], data_name)
draw_seed(axs[1, 1], seed_dic[data_name][3],seed_dic[data_name][3], data_name)

# 共享图例
lines, labels = axs[0,0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.52, 1.12), fancybox=True, mode='normal', ncol=4, markerscale = 1.3, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Graphs',data_name) + '_seed.pdf', bbox_inches='tight')