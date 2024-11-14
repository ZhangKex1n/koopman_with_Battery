import torch
import koopmanlab as kp
import sys
from data_mat_helper import load_battery_data
import numpy as np
# Setting your computing device
torch.cuda.set_device(0)
device = torch.device("cuda")
sys.stdout.flush()

# Path
# data_path = "./data/ns_V1e-3_N5000_T50.mat"
# fig_path = "./demo/fig/"
# save_path = "./demo/result/"
fig_path=r'C:\Users\Crisy\Desktop'
# file_path = r'C:\Users\Crisy\Desktop\Matlab_Common_PredictionModel\B0038.mat'
# data = load_battery_data(file_path)
# Voltage_measured_list=data['24']['Voltage_measured']
# flattened = [arr.flatten() for arr in Voltage_measured_list]
# Voltage_measured_list = np.concatenate(flattened)
# Loading Data
train_loader, test_loader = kp.data.prepare_battery_data(batch_size=5, T_in=70, T_out=30, sub=1)
# Hyper parameters
ep = 1# Training Epoch
o = 70 # Koopman Operator Size
m = 16 # Modes
r = 100 # Power of Koopman Matrix #koopman训练次数

# Model
koopman_model = kp.model.koopman(backbone = "KNO1d", autoencoder = "test", o = o, m = m, r = r, t_in = 70, device = device)
koopman_model.compile()
koopman_model.opt_init("Adam", lr = 0.001, step_size=5, gamma=0.5)
koopman_model.train(epochs=ep, trainloader = train_loader)

# Result and Saving
time_error = koopman_model.test(test_loader, path = fig_path, is_save = True, is_plot = True)
#filename = "ns_time_error_op" + str(o) + "m" + str(m) + "r" +str(r) + ".pt"
#torch.save({"time_error":time_error,"params":koopman_model.params}, save_path + filename)
