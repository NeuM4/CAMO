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
Dic = {'AR_UCB': ['#000080', "^", "AR-MF-UCB", "solid"],
       'AR_EI': ['#000080', "s", "AR-MF-EI", "solid"],
       'AR_cfKG': ['#000080', "X", "AR-cfKG", "solid"],
       'ResGP_UCB': ['#00CCFF', "^", "ResGP-MF-UCB", "solid"],
       'ResGP_EI': ['#00CCFF', "s", "ResGP-MF-EI", "solid"],
       'ResGP_cfKG': ['#00CCFF', "X", "ResGP-cfKG", "solid"],
        
        'DNN_MFBO': ['#228B22', "*", "DNN", "solid"],
        
        # 'GP_UCB': ['#4169E1', "^", "MF-GP-UCB", "solid"],
        # 'GP_EI': ['#4169E1', "s", "GP-MF-EI", "solid"],
        # 'GP_cfKG': ['#4169E1', "X", "cfKG", "solid"],
        
        'GP_UCB': ['#4169E1', "^", "BOCA", "solid"],
        'GP_cfKG': ['#4169E1', "X", "cfKG", "solid"],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAMO-BOCA", "dashed"], # red
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", "dashed"],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-BOCA", "dashed"], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", "dashed"],
        
        'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac':['#006400', "*", "SMAC3", 'solid'],
        }



max_dic = {'Branin': 55,'Currin': 14,'Park': 2.2, 'VibratePlate': 250, 'HeatedBlock': 2,'bohachevsky': 72.15,'borehole':0}
add_dict = {'Branin': 3,'Currin': 0.01,'Park': 0.3, 'VibratePlate': 0, 'HeatedBlock': 0,'bohachevsky': 4,'borehole':0}
cost_lim_y = {'Branin': [2,12], 'Currin': [0, 1.75], 'Park': [0.2, 1.4],'bohachevsky': [0, 32], 'borehole': [0, 0.5]}
cost_lim_x = {'Branin': [13, 150], 'Currin': [13, 150], 'Park': [13, 150],'bohachevsky': [13, 150], 'borehole': [13, 150]}


##Branin
# data_name = 'Branin'
# seed_dic ={'pow_10':[2,4,5,7,8],'linear':[4,5,7,8],'log':[2,3,5,7,8]}
##Currin
# data_name = 'Currin'
# seed_dic ={'pow_10':[2,3,4,5,6,7,8,9],'linear':[0,2,3,4,5,6,7,8],'log':[0,2,3,4,5,7,8,9]}
##Park
# data_name = 'Park'
# seed_dic ={'pow_10':[0,2,4,5,7],'linear':[0,4,7,9],'log':[0,1,2,3,4,7]}

##bohachevsky
data_name = 'bohachevsky'
seed_dic ={'pow_10':[1,17,19,21,22],'linear':[1,17,19,21,22],'log':[1,17,19,21,22]}

##
# data_name = 'borehole'
# seed_dic ={'pow_10':[0,1,2,6],'linear':[0,1,2,6],'log':[0,1,2,6]}

methods_name_list = [ 
                     'GP_UCB', 
                     'GP_cfKG',
                     'fabolas','smac',
                     'CMF_CAR_UCB',
                    #  'CMF_CAR_cfKG',
                        
                     'CMF_CAR_dkl_UCB',
                    #  'CMF_CAR_dkl_cfKG'
                    ]
baseline_list = ['fabolas','smac']

cost_list = ['log', 'linear', 'pow_10']
cost_label_dic = {'log': 'Log', 'linear': 'Linear', 'pow_10': 'Power 10'}
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

Exp_marker = "bohachevsky_30seed"

for kk in range(3):
    cost_name = cost_list[kk]
    for methods_name in methods_name_list:
        print(methods_name)
        cost_collection = []
        # SR_collection = []
        inter_collection = []
        for seed in seed_dic[cost_name]:
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            # cost = data['cost'].to_numpy().reshape(-1, 1)
            # SR = data['SR'].to_numpy().reshape(-1, 1)
            cost = data['cost'].to_numpy()
            
            if methods_name == 'fabolas':
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy()
            # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            # SR_collection.append(SR)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        if  cost_name == 'log':
            if methods_name in ['CMF_CAR_UCB','CMF_CAR_cfKG','CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']:
                makervery_index = 90
            else:
                makervery_index = 90
        elif  cost_name == 'linear':
            makervery_index = 20
        else:
            makervery_index = 14
        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=makervery_index)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    # for methods_name in baseline_list:
    #     print(methods_name)
    #     cost_collection = []
    #     # SR_collection = []
    #     inter_collection = []
    #     for seed in [1,17,19,21,22]:
    #         path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results',Exp_marker,
    #                             data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
    #         data = pd.DataFrame(pd.read_csv(path))
    #         # cost = data['cost'].to_numpy().reshape(-1, 1)
    #         # SR = data['SR'].to_numpy().reshape(-1, 1)
    #         cost = data['cost'].to_numpy()
            
    #         if methods_name == 'fabolas':
    #             SR = max_dic[data_name] - data['incumbents'].to_numpy()
    #         else:
    #             SR = data['SR'].to_numpy()
    #         inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
    #         # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
    #         # SR_collection.append(SR)
    #         cost_collection.append(cost)
    #         inter_collection.append(inter)

    #     cost_x = np.unique(np.concatenate(cost_collection))
    #     SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
    #     SR_new = np.array(SR_new)
    #     mean = np.mean(SR_new, axis=0)
    #     var = np.std(SR_new, axis=0)
    #     if  cost_name == 'log':
    #         if methods_name in ['CMF_CAR_UCB','CMF_CAR_cfKG','CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']:
    #             makervery_index = 90
    #         else:
    #             makervery_index = 90
    #     elif  cost_name == 'linear':
    #         makervery_index = 20
    #     else:
    #         makervery_index = 14
    #     ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
    #                 label=Dic[methods_name][2],
    #                 marker=Dic[methods_name][1], markersize=12, markevery=makervery_index)
    #     axs[kk].fill_between(cost_x,
    #                     mean + add_dict[data_name] - 0.96 * var,
    #                     mean + add_dict[data_name] + 0.96 * var,
    #                     alpha=0.05, color=Dic[methods_name][0])
        
    # plt.yscale('log')
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
# lines, labels = axs[0].get_legend_handles_labels()
# leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.13), fancybox=True, mode='normal', ncol=6, markerscale = 1.3, fontsize=25)

# change the line width for the legend
# for line in leg.get_lines():
#     line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'ICLR_res') + '/' +'CMF_' + data_name + '_cost.pdf', bbox_inches='tight')
