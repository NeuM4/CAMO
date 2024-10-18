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
        
        'GP_UCB': ['#4169E1', "^", "MF-GP-UCB", "solid"],
        'GP_EI': ['#4169E1', "s", "GP-MF-EI", "solid"],
        'GP_cfKG': ['#4169E1', "X", "cfKG", "solid"],
        'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB", "solid"],
        'GP_cfKG_con': ['#4169E1', "X", "Con_GP-cfKG", "solid"],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAMO-BOCA", "dashed"], # red
        'CMF_CAR_EI':['#FF0000',"s","CAMO-EI", "dashed"],
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", "dashed"],

        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-BOCA", "dashed"], # orange
        'CMF_CAR_dkl_EI': ['#FF5E00',"s","CAMO-DKL-EI","dashed"],
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", "dashed"],

        'DMF_CAR_UCB':['black',"^","DMF_CAR-BOCA","dashed"],
        'DMF_CAR_EI':['black',"s","DMF_CAR-EI","dashed"],
        'DMF_CAR_cfKG':['black',"X","DMF_CAR-cfKG","dashed"],
        
        'DMF_CAR_dkl_UCB':['green',"^","DMF_CAR_dkl-BOCA","dashed"],
        'DMF_CAR_dkl_EI':['green',"s","DMF_CAR_dkl-EI","dashed"],
        'DMF_CAR_dkl_cfKG':['green',"X","DMF_CAR_dkl-cfKG","dashed"],

        }


max_dic = {'non_linear_sin':0, 'forrester': 50,'Branin': 55,'Currin': 14,'Park': 2.2}
add_dict = {'forrester': 0 ,'non_linear_sin': 0,'Branin': 0.85,'Currin': 0.01,'Park': 0.1, 'VibratePlate': 0, 'HeatedBlock': 1.2}
lim_x = {'forrester': [48, 135], 'non_linear_sin': [48, 150],
         'Branin':[90,150],'Currin':[90,150],'Park':[90,150]}
lim_y = {'forrester': [0, 52], 'non_linear_sin': [0.007, 0.034],'Currin':[0,1.8],'Park':[0,1.2],'Branin':[0,9]}

methods_name_list = [
                    'AR_UCB', 'ResGP_UCB','GP_UCB',
                    'AR_EI',  'ResGP_EI','GP_EI',
                    'AR_cfKG', 'ResGP_cfKG','GP_cfKG',
                    #  'CMF_CAR_EI',
                    #  'CMF_CAR_dkl_EI',
                     ]

our_method_list =   [
                        'CMF_CAR_UCB',
                     'CMF_CAR_dkl_UCB',
                     'CMF_CAR_cfKG',
                     'CMF_CAR_dkl_cfKG',
                     ]
baseline_list = ['DNN_MFBO']
seed_dict = {'Branin':[0,1,2,3,5,6,7,8,9],'Currin':[0,3,4,5],"Park":[],'forrester':[0,4,5,9],'non_linear_sin':[0,1,3,7,8]}

data_list = ['Branin', 'Currin']
Exp_marker = "Norm_res"
cost_name = 'pow_10'
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
for kk in range(2):
    data_name = data_list[kk]
    for methods_name in methods_name_list:
        cost_collection = []
        # SR_collection = []
        inter_collection = []
        for seed in seed_dict[data_name]:
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results', Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            # cost = data['cost'].to_numpy().reshape(-1, 1)
            # SR = data['SR'].to_numpy().reshape(-1, 1)
            cost = data['cost'].to_numpy()
            
            SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            # SR_collection.append(SR)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)

        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=14)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
    for methods_name in our_method_list:
        print(methods_name)
        cost_collection = []
        inter_collection = []
        for seed in seed_dict[data_name]:
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results','improve_our_con-dis',
                                data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
            data = pd.read_csv(path)
            cost = data['cost'].to_numpy()
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

        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=14)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
            
    for methods_name in baseline_list:
        print(methods_name)
        cost_collection = []
        inter_collection = []
        for seed in [0,1]:
            path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
            data = pd.read_csv(path)
            cost = data['cost'].to_numpy()
            SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            cost_collection.append(cost)
            inter_collection.append(inter)
        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)

        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=14)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
            

    # axs[kk].set_yscale('log')
    axs[kk].set_xlabel("Cost", fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    axs[kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()

# label = [Dic[i][-1] for i in methods_name_list]
# label = label + [Dic[i+'_con'][-1] for i in cmf_methods_name_list]

# 共享图例
lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.51, 1.32), fancybox=True, mode='normal', ncol=5, markerscale = 1.3, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'ICLR_graphs') + '/' +'Con-dis-' + Exp_marker + '_SR_DNN.pdf', bbox_inches='tight')