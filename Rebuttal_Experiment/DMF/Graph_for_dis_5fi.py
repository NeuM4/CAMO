import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


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
seed_dict = {'Branin':[0,1,6,8,9]}


# methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG', 'GP_UCB', 'GP_EI', 'GP_cfKG',
#                      'DMF_CAR_UCB','DMF_CAR_EI','DMF_CAR_cfKG','DMF_CAR_dkl_UCB','DMF_CAR_dkl_EI','DMF_CAR_dkl_cfKG',
#                      'CMF_CAR_UCB','CMF_CAR_EI','CMF_CAR_cfKG','CMF_CAR_dkl_UCB','CMF_CAR_dkl_EI','CMF_CAR_dkl_cfKG',
#                      ]

methods_name_list = [
                    'AR_UCB', 'ResGP_UCB','GP_UCB',
                    'AR_EI',  'ResGP_EI','GP_EI',
                    'AR_cfKG', 'ResGP_cfKG','GP_cfKG',
                     'CMF_CAR_EI','CMF_CAR_dkl_EI',
                     ]

our_method_list =   [
                    # 'CMF_CAR_UCB',
                     'CMF_CAR_dkl_UCB',
                     'CMF_CAR_cfKG',
                     'CMF_CAR_dkl_cfKG',
                     ]
baseline_list = ['DNN_MFBO']

# data_list = ['Forrester']
cost_list = ['pow_10']
cost_label_dic = {'log': 'Log', 'linear': 'Linear', 'pow_10': 'Power 10'}
fig, ax = plt.subplots(figsize=(14, 6))

data_name = 'Park'  # 将 data_name 提升到外部，因为它在循环中没有变化
Exp_marker = "Con_dis_five_fidelity"
cost_name = 'pow_10'

for methods_name in methods_name_list:
    print(methods_name)
    cost_collection = []
    inter_collection = []
    for seed in [0,1,2,3,4,5,6,7,8]:
        path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results',Exp_marker,
                            data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
        # path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results',
        #                     data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
        data = pd.read_csv(path)
        cost = data['cost'].to_numpy()
        SR = data['SR'].to_numpy()
        # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
        inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
        cost_collection.append(cost)
        inter_collection.append(inter)

    cost_x = np.unique(np.concatenate(cost_collection))
    SR_new = np.array([inter(cost_x) for inter in inter_collection])
    mean = np.mean(SR_new, axis=0)
    var = np.std(SR_new, axis=0)

    if cost_name == 'log':
        makervery_index = 90 if methods_name in ['CMF_CAR_UCB', 'CMF_CAR_cfKG', 'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG'] else 90
    elif cost_name == 'linear':
        makervery_index = 20
    else:
        makervery_index = 14

    ax.plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
            label=Dic[methods_name][2], marker=Dic[methods_name][1], markersize=12, markevery=makervery_index)
    ax.fill_between(cost_x, mean + add_dict[data_name] - 0.96 * var,
                    mean + add_dict[data_name] + 0.96 * var, alpha=0.05, color=Dic[methods_name][0])

for methods_name in our_method_list:
    print(methods_name)
    cost_collection = []
    inter_collection = []
    for seed in [0,1,2,3,4,5,6,7,8]:
        path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results', Exp_marker ,
                            data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
        # path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results',
        #                     data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
        data = pd.read_csv(path)
        cost = data['cost'].to_numpy()
        SR = data['SR'].to_numpy()
        # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
        inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
        cost_collection.append(cost)
        inter_collection.append(inter)

    cost_x = np.unique(np.concatenate(cost_collection))
    SR_new = np.array([inter(cost_x) for inter in inter_collection])
    mean = np.mean(SR_new, axis=0)
    var = np.std(SR_new, axis=0)

    if cost_name == 'log':
        makervery_index = 90 if methods_name in ['CMF_CAR_UCB', 'CMF_CAR_cfKG', 'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG'] else 90
    elif cost_name == 'linear':
        makervery_index = 20
    else:
        makervery_index = 14

    ax.plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
            label=Dic[methods_name][2], marker=Dic[methods_name][1], markersize=12, markevery=makervery_index)
    ax.fill_between(cost_x, mean + add_dict[data_name] - 0.96 * var,
                    mean + add_dict[data_name] + 0.96 * var, alpha=0.05, color=Dic[methods_name][0])


ax.set_xlabel("Cost", fontsize=25)
ax.set_ylabel("Simple regret", fontsize=25)
# ax.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
# ax.set_yscale('log')
# ax.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
ax.tick_params(axis='both', labelsize=20)
ax.grid()
# ax.legend(fontsize=15)
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=15)

plt.tight_layout()  # 调整布局，以防止标签重叠
# plt.show()
# plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Graphs') + '/' +'Con-dis-' + Exp_marker+ data_name + '_SR_DNN2.pdf', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Graphs') + '/' +'Dis_fivefi_' + data_name + '_SR2.pdf', bbox_inches='tight')
