import numpy as np
from scipy.io import loadmat
def truncate_and_convert_to_array(data, max_length=104):
    # 确保所有序列都被截断到最大长度
    truncated_data = [seq[:max_length] for seq in data]
    # 转换为NumPy数组
    return np.array(truncated_data)

def load_battery_data(file_path):
    # 加载MAT文件
    data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    cycles = data['B0038'].cycle  # 'B0038'是struct的名称，'cycle'是内部的字段

    # 准备数据容器
    discharge_data = {
        '24': {'Voltage_measured':[], 'Current_load': [], 'Voltage_load': [], 'Time': [], 'Capacity': []},
        '44': {'Voltage_measured':[], 'Current_load': [], 'Voltage_load': [], 'Time': [], 'Capacity': []}
    }

    # 筛选数据
    for cycletemp in cycles:
        if cycletemp.type == 'discharge':  # 筛选放电数据
            temperature_key = '24' if cycletemp.ambient_temperature == 24 else '44'
            discharge_data[temperature_key]['Voltage_measured'].append(cycletemp.data.Voltage_measured)
            discharge_data[temperature_key]['Current_load'].append(cycletemp.data.Current_load)
            discharge_data[temperature_key]['Voltage_load'].append(cycletemp.data.Voltage_load)
            discharge_data[temperature_key]['Time'].append(cycletemp.data.Time)
            discharge_data[temperature_key]['Capacity'].append(cycletemp.data.Capacity)

        #print("cycle.data.Capacity")
    # # 转换列表为NumPy数组并截断
    for temp in discharge_data:
        for key in ['Current_load', 'Voltage_load', 'Time']:
            discharge_data[temp][key] = truncate_and_convert_to_array(discharge_data[temp][key])
        discharge_data[temp]['Capacity'] = np.array(discharge_data[temp]['Capacity'])

    return discharge_data


# def prepare_train_test_data(file_path):
# # 加载数据
file_path = r'C:\Users\Crisy\Desktop\Matlab_Common_PredictionModel\B0038.mat'
data = load_battery_data(file_path)
# print("aaaaaaa")
#
# # 数据划分为训练集和测试集
#     train_data, test_data = {}, {}
#     for temp in data:
#         X = np.stack((data[temp]['Current_load'], data[temp]['Voltage_load']), axis=-1)  # 将电流和电压堆叠
#         y = data[temp]['Capacity']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     #指定测试比例20% random_state=42 是一个随机种子，用于确保 train_test_split 函数每次运行时生成相同的训练集和测试集。通过设置这个值，可以使数据划分过程具有可重复性，以便进行相同的实验或调试。
#         train_data[temp] = {'X': X_train, 'y': y_train}
#         test_data[temp] = {'X': X_test, 'y': y_test}
#     return train_data,test_data
# # 接下来可以使用train_data和test_data来训练和测试Koopman模型
