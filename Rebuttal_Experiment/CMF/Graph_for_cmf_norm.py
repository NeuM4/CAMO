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
        'AR_UCB': ['#000080', "^", "AR_UCB"],
        'AR_EI': ['#000080', "s", "AR_EI"],
        'AR_cfKG': ['#000080', "o", "AR_cfKG"],
        'ResGP_UCB': ['#00CCFF', "^", "ResGP_UCB"],
        'ResGP_EI': ['#00CCFF', "s", "ResGP_EI"],
        'ResGP_cfKG': ['#00CCFF', "o", "ResGP_cfKG"],
        
        'DNN_MFBO': ['#228B22', "X", "DNN"],
        'fabolas':['#808000', "*", "Fabolas","-"],
        'smac':['#006400', "*", "SMAC3",'-'],
        
        'GP_UCB': ['#4169E1', "^", "GP_UCB","dashed"],
        'GP_cfKG': ['#4169E1', "o", "GP_cfKG","dashed"],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAR_UCB","dashed"], # red
        'CMF_CAR_cfKG': ['#FF0000', "o", "CAR_cfKG","dashed"],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAR_dkl_UCB","dashed"], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "o", "CAR_dkl_cfKG","dashed"],
        }


data_name = 'borehole'
# data_name = 'VibratePlate'
# data_name = 'Branin'
# data_name = 'Currin'
# Exp_marker = "eight_dim_exp"
Exp_marker = "borehole_improve"

max_dic = {'non_linear_sin':0, 'forrester': 50,'Branin': 55,'Currin': 14,'Park': 2.2,'colvile':609.26,'himmelblau':303.5,'bohachevsky':72.15,'borehole':244}
add_dict = {'forrester': 0 ,'non_linear_sin': 0,'Branin': 0.85,'Currin': 0.01,'Park': 0.1, 
            'VibratePlate': 0, 'HeatedBlock': 1.2, 'borehole':0,'booth':0,'hartmann':0.0001,"bohachevsky":4,'himmelblau':1.5,'colvile':125}
## pow_10
cost_name = 'pow_10'
lim_x = {'forrester': [48, 135], 'non_linear_sin': [48, 150],
         'Branin':[48,150],'Currin':[48,150],'Park':[48,150],'VibratePlate':[48,150],'HeatedBlock':[48,150],
         'borehole':[48,150],'booth':[48,150],'hartmann':[48,150],"bohachevsky":[48,150],'himmelblau':[48,150],'colvile':[48,150]}
## linear
# cost_name = 'linear'
# lim_x = {'Forrester': [48, 135], 'non_linear_sin': [48, 150],
#          'Branin':[30,150],'Currin':[30,150],'Park':[30,150],'VibratePlate':[48,150],'HeatedBlock':[48,150],'borehole':[30,150],
#          'booth':[30,150],'hartmann':[30,150],"bohachevsky":[30,150],'himmelblau':[30,150]}
## log
# cost_name = 'log'
# lim_x = {'Forrester': [48, 135], 'non_linear_sin': [48, 150],
#          'Branin':[16,150],'Currin':[16,150],'Park':[16,150],'VibratePlate':[48,150],
#          'HeatedBlock':[48,150],'borehole':[16,150],'booth':[16,150],'hartmann':[16,150],"bohachevsky":[16,150],'himmelblau':[16,150]}

lim_y = {'forrester': [0, 40], 'non_linear_sin': [0.007, 0.034],'Currin':[0,2],'Park':[0,1.2],'Branin':[0,10],'bohachevsky':[0,35],
         'colvile':[0,420],'himmelblau':[0,148]}

methods_name_list = [
                    'GP_UCB', 
                    'GP_cfKG',
                    'CMF_CAR_UCB',
                    # 'CMF_CAR_cfKG',
                     'CMF_CAR_dkl_UCB', 
                    #  'CMF_CAR_dkl_cfKG',
                    'fabolas',
                    'smac'
                         ]

line = []
fig, ax = plt.subplots(figsize=(14, 6))
for methods_name in methods_name_list:
    print(methods_name)
    cost_collection = []
    inter_collection = []
    for seed in [1,2,7,12,17]:
        path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results',Exp_marker,
                            data_name, cost_name, f'{methods_name}_seed_{seed}.csv')
        data = pd.read_csv(path)
        cost = data['cost'].to_numpy()
        if methods_name == 'fabolas':
            SR = max_dic[data_name] - data['incumbents'].to_numpy()
        else:
            SR = data['SR'].to_numpy()
        inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
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

# ax.set_yscale('log')
ax.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
# ax.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
ax.tick_params(axis='both', labelsize=15)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=20)
ax.grid()
plt.tight_layout()
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' + data_name +'_'+ cost_name +'_SR_Interpolation.png', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graphs') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_SR'+'.pdf', bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_seed') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_seed'+str(seed)+'_.eps', bbox_inches='tight')
