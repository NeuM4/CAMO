import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def read_txt_file(file_path, data_name, max_value, seed):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            recording = {'cost':[], 'SR': [], 'operation_time': []}
            for line in file:
                content = line.strip()
                if content[:19] == '* simple optimum = ':
                    recording['SR'].append(max_value - float(content[20:-1]))
                elif content[:15] == '* cost so far: ':
                    recording['cost'].append(float(content[15:]))
                elif content[:15] == "* time spent = ":
                    recording['operation_time'].append(float(content[15:]))
            df = pd.DataFrame(recording)
            # df.to_csv(sys.path[-1] + '/Exp_results_time/' + data_to_name + '/pow_10/DNN_MFBO_seed_' + str(seed-1) + '.csv', index=False)
            df.to_csv(sys.path[-1] + '/Exp_results/Baseline/' + f'DNN_MFBO_seed_' + str(seed-1) + '.csv', index=False)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 指定要读取的txt文件的路径
    file_path = sys.path[-1]
    data_name = 'Currin'
    max_dict = {'Non_linear_sin': 0.33, 'forrester': 48.51,'Branin': 55,'Currin': 14,'Park': 2.2}
    for seed in [1,2]:
        read_txt_file(file_path + '\Exp_results\Baseline\log-' + data_name +'_Condis_'+ str(seed) + '.txt', data_name, max_dict[data_name], seed)
