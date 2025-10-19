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


add_dic = {'VibratePlate': 0, 'HeatedBlock': 1}
max_dic = {'VibratePlate': 250, 'HeatedBlock': 2}
lim_x = {'VibratePlate': [48, 300], 'HeatedBlock': [48, 300]}
lim_y = {'VibratePlate': [28, 41.8], 'HeatedBlock': [0,1.44]}

seed_dic = {'VibratePlate': [i for i in range(30)], 
            'HeatedBlock': [i for i in range(30)]}

cmf_methods_name_list = [
                         'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB','CMF_CAR_cfKG',
                         ]
baseline_list = [
    'fabolas',
    'smac']

data_list = ['VibratePlate', 'HeatedBlock']
cost_name = 'pow_10'
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            if data_name == 'HeatedBlock':
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            else:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
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
        

        ll = axs[0, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=60,linewidth=2)
        
        axs[0,kk].text(0.5, 1.02, data_name, transform=axs[0,kk].transAxes, ha='center', fontsize=25)
        
    for methods_name in baseline_list:
        cost_collection = []
        inter_collection = []
    
        for seed in seed_dic[data_name]:
            
            if data_name == 'HeatedBlock':
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            else:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            if methods_name == 'fabolas':
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        
        methods_name = methods_name

        ll = axs[0, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=60,linewidth=2)
        
        axs[0,kk].text(0.5, 1.02, data_name, transform=axs[0,kk].transAxes, ha='center', fontsize=25)
        
    axs[0, kk].set_xlabel("Cost", fontsize=25)
    axs[0, kk].set_ylabel("Simple regret", fontsize=25)
    # axs[0, kk].set_yscale('log')
    axs[0, kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[0, kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[0, kk].tick_params(axis='both', labelsize=20)
    axs[0, kk].grid()
    
    

# lim_x = {'VibratePlate': [80, 350], 'HeatedBlock': [22, 200]}
for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            if data_name == 'HeatedBlock':
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            else:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost_tem = data['operation_time'].to_numpy()
            cost = np.cumsum(cost_tem)
            if methods_name in ['fabolas']:
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            else:
                SR = data['SR'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        

        ll = axs[1, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=300)
    
    for methods_name in baseline_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            if data_name == 'HeatedBlock':
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            else:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results','Norm_res',
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost_tem = data['operation_time'].to_numpy()
            cost = np.cumsum(cost_tem)
            if methods_name in ['fabolas']:
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            else:
                SR = data['SR'].to_numpy()
                if methods_name == 'smac' and data_name == 'VibratePlate':
                    cost = cost+106
                inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            
            # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)

        ll = axs[1, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=100)
        
        
    axs[1, kk].set_xlabel("Wall clock time (s)", fontsize=25)
    axs[1, kk].set_ylabel("Simple regret", fontsize=25)
    axs[1, kk].set_xscale('log')
    
    axs[1, kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[1, kk].tick_params(axis='both', labelsize=20)
    axs[1, kk].grid()


lines, labels = axs[1, 1].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.11), fancybox=True, mode='normal', ncol=6, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Graphs') + '/' + 'CMF_real_' + cost_name +'_SR_together_4.png', bbox_inches='tight')