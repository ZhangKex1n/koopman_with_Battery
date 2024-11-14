import torch
import koopmanlab as kp
# Setting your computing device
torch.cuda.set_device(0)
device = torch.device("cuda")

# Path
data_path = "./data/ns_V1e-3_N5000_T50.mat"
fig_path = "./demo/fig/"
save_path = "./demo/result/"

# Loading Data
train_loader, test_loader = kp.data.navier_stokes(data_path, batch_size = 10, T_in = 10, T_out = 40, type = "1e-3", sub = 1)


# Hyper parameters
ep = 40 # Training Epoch 学习次数
o = 32 # Koopman Operator Size
m = 16 # Modes
r = 8 # Power of Koopman Matrix

# Model
koopman_model = kp.model.koopman(backbone = "KNO2d", autoencoder = "MLP", o = o, m = m, r = r, t_in = 10, device = device) #小的t_in要和大T_in对应
koopman_model.compile()
koopman_model.opt_init("Adam", lr = 0.0001, step_size=100, gamma=0.5) #学习率太大1
koopman_model.train(epochs=ep, trainloader = train_loader, evalloader = test_loader)

# Result and Saving
time_error = koopman_model.test(test_loader, path = fig_path, is_save = True, is_plot = True)
# filename = "ns_time_error_op" + str(o) + "m" + str(m) + "r" +str(r) + ".pt"
# torch.save({"time_error":time_error,"params":koopman_model.params}, save_path + filename)
