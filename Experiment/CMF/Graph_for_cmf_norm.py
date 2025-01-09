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
        'fabolas':['#808000', "*", "Fabolas","solid"],
        'smac':['#006400', "*", "SMAC3",'solid'],
        
        'GP_UCB': ['#4169E1', "^", "GP_UCB","dashed"],
        'GP_cfKG': ['#4169E1', "o", "GP_cfKG","dashed"],
        'GP_dkl_UCB': ['green', "^", "GP_dkl_UCB","dashed"],
        'GP_dkl_cfKG': ['green', "o", "GP_dkl_cfKG","dashed"],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAR_UCB","dashed"], # red
        'CMF_CAR_cfKG': ['#FF0000', "o", "CAR_cfKG","dashed"],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAR_dkl_UCB","dashed"], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "o", "CAR_dkl_cfKG","dashed"],
        }


# data_name = 'borehole'
# data_name = 'VibratePlate'
# data_name = 'Branin'
data_name = 'colvile'
# Exp_marker = "eight_dim_exp"
Exp_marker = "Norm_res_60"

max_dic = {'forrester': 50, 'non_linear_sin':0.033,'Branin': 55,'Currin': 14,'Park': 2.2,'himmelblau':303.5,'bohachevsky': 72.15}
# add_dict = {'forrester': 0 ,'non_linear_sin': 0,'Branin': 0.85,'Currin': 0.01,'Park': 0.1, 
#             'VibratePlate': 0, 'HeatedBlock': 1.2, 'borehole':0,'booth':0,'hartmann':0.0001,"bohachevsky":4,'himmelblau':1.5,'colvile':125}
add_dict = {'forrester': 0.8 , 'non_linear_sin': 0.1,'Branin': 0.86,'Currin': 0.01,'Park': 0.1, 'himmelblau': 1,'bohachevsky': 4,'VibratePlate': 0, 'HeatedBlock': 1.2,'borehole':0,'colvile':125}
## pow_10
cost_name = 'pow_10'
lim_x = {'forrester': [48, 300], 'non_linear_sin': [48, 300],
         'Branin':[48,300],'Currin':[48,300],'Park':[48,300],'VibratePlate':[48,150],'HeatedBlock':[48,150],
         'borehole':[48,300],'booth':[48,150],'hartmann':[48,150],"bohachevsky":[48,300],'himmelblau':[48,300],'colvile':[48,300]}
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

lim_y = {'forrester': [0, 52], 'non_linear_sin': [0,0.035], 'Branin':[0,10], 'Currin':[0,1.75],'Park':[0,1.2],'himmelblau':[0, 150],'bohachevsky':[0,32]}
# seed_dic = {'forrester': [0,1,2,3,5,9], 'non_linear_sin': [1,4,5,6,9], 'Branin':[2,4,5,7,8], 'Currin':[2,3,4,5,6,7,8],'Park':[0,2,4,5,7],'himmelblau':[0,1,2,3,8],
#             'bohachevsky':[1,17,19,21,22]}
seeds = list(range(0, 30))  # 生成 1 �? 30 的列�?
# exclude_dict = {'Park':{13, 15, 18},'non_linear_sin':{0,2,3,4,8,9,10,23,28},'bohachevsky':{6,8,16,23,28,29}}      # 定义需要去除的值，使用集合以加快查找速度
# seed_dic ={'Branin':list(range(0, 30)),'Currin':list(range(2,29)),'Park':[s for s in seeds if s not in exclude_dict['Park']],'non_linear_sin':[5,6,7,17,18,25,26,27],
#            'bohachevsky':[s for s in seeds if s not in exclude_dict['bohachevsky']],'forrester':[1,4,5,7,8,9,11,13,14,16,18,19,22,23,24,26,27]}
seed_dic ={'Currin':[range(0, 30)],'Branin':[range(0, 30)],'Park':[range(0, 30)],
           'non_linear_sin':[range(0, 30)],'forrester':[range(0, 30)],
           'bohachevsky':[range(0, 30)],'himmelblau':[range(0, 30)],
           'borehole':[range(0, 30)],'colvile':[range(0, 30)]}

methods_name_list = [
                    'GP_UCB', 
                    # 'GP_dkl_UCB',
                    # 'GP_cfKG',
                    # 'GP_dkl_cfKG',
                    'CMF_CAR_UCB',
                    # 'CMF_CAR_cfKG',
                     'CMF_CAR_dkl_UCB', 
                    #  'CMF_CAR_dkl_cfKG',
                    # 'fabolas',
                    # 'smac'
                         ]

line = []
fig, ax = plt.subplots(figsize=(14, 6))
label_name = []
for methods_name in methods_name_list:
    print(methods_name)
    cost_collection = []
    inter_collection = []
    smac_dict = [range(0, 30)]
    if methods_name in ['smac']:
        for seed in smac_dict:
            path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            cost_collection.append(cost)
            inter_collection.append(inter)
    else:
        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            if methods_name in ['fabolas', 'smac']:
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
    # if methods_name in ['fabolas', 'smac']:
    var = np.std(SR_new, axis=0)
    # else:
    #     mean = np.insert(mean, 0, opt_dic[data_name])
    #     cost_x = np.insert(cost_x, 0, 50)
    #     var = np.std(SR_new, axis=0)
    #     var = np.insert(var, 0, 0.5)

    new_method_name = methods_name
    label_name.append(new_method_name)
    
    ax.plot(cost_x, mean + add_dict[data_name], ls=Dic[new_method_name][-1], color=Dic[new_method_name][0],
                label=Dic[new_method_name][2],
                marker=Dic[new_method_name][1], markersize=12, markevery=17)
    
    # markevery_indices = range(0, len(cost_x), 17)
    # errorbar_x = [cost_x[i] for i in markevery_indices]
    # errorbar_y = [mean[i] + add_dict[data_name] for i in markevery_indices]
    # errorbar_yerr = [0.96 * var[i] for i in markevery_indices]

    # ax.errorbar(
    #     errorbar_x, 
    #     errorbar_y, 
    #     yerr=errorbar_yerr, 
    #     fmt=Dic[new_method_name][1],  # 标记样式
    #     markersize=12, 
    #     color=Dic[new_method_name][0], 
    #     capsize=5,  # 误差条端点长�??
    #     alpha=0.8  # 线条透明�??
    #     )
    ax.fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])

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
plt.savefig(os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Graphs') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_SR_dkl'+'.pdf', bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_seed') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_seed'+str(seed)+'_.eps', bbox_inches='tight')
