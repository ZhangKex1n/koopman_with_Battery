import h5py
import torch
import scipy.io
import numpy as np
from data_mat_helper import load_battery_data

def burgers(path, batch_size = 64, sub = 32):
    f = scipy.io.loadmat(path)
    x_data = f['a'][:,::sub]
    y_data = f['u'][:,::sub]

    x_train = torch.tensor(x_data[:1000,:],dtype=torch.float32)
    y_train = torch.tensor(y_data[:1000,:],dtype=torch.float32)
    x_test = torch.tensor(x_data[-200:,:],dtype=torch.float32)
    y_test = torch.tensor(y_data[-200:,:],dtype=torch.float32)

    S = x_train.shape[1]

    x_train = x_train.reshape(1000,S,1)
    x_test = x_test.reshape(200,S,1)
    x_test = x_test
    y_test = y_test

    print("Burgers Dataset has been loaded successfully!")
    print("X train shape:", x_train.shape, "Y train shape:", y_train.shape)
    print("X test shape:", x_test.shape, "Y test shape:", y_test.shape)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def shallow_water(path, batch_size = 20, T_in = 10, T_out = 20, sub = 1):
    ntrain = 900
    ntest = 100
    total = ntrain + ntest
    f = h5py.File(path)
    data = f['data'][0:total]
    data = torch.tensor(data,dtype=torch.float32)
    # Traning data
    train_a = data[:ntrain,::sub,::sub,:T_in]
    train_u = data[:ntrain,::sub,::sub,T_in:T_out+T_in]
    # Testing data
    test_a = data[-ntest:,::sub,::sub,:T_in]
    test_u = data[-ntest:,::sub,::sub,T_in:T_out+T_in]
    
    print("Shallow Water Equations Dataset has been loaded successfully!")
    print("X train shape:", train_a.shape, "Y train shape:", train_u.shape)
    print("X test shape:", test_a.shape, "Y test shape:", test_u.shape)
        
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    
def navier_stokes(path, batch_size = 20, T_in = 10, T_out = 40, type = "1e-3", sub = 1,reshape = False):
    if type == "1e-3":
        ntrain = 1000
        ntest = 200
        total = ntrain + ntest
        f = h5py.File(path)
        data = f['u'][...,0:total]
        print("dataset shape : ", data.shape) # Print original shape of the data
        data = torch.tensor(data,dtype=torch.float32)
        data = data.permute(3,1,2,0) # The dimension of the data shape is [B, X, Y, T]

        # Traning data
        train_a = data[:ntrain,::sub,::sub,:T_in] #特征1
        train_u = data[:ntrain,::sub,::sub,T_in:T_out+T_in]
        # Testing data
        test_a = data[-ntest:,::sub,::sub,:T_in]
        test_u = data[-ntest:,::sub,::sub,T_in:T_out+T_in]

        if reshape:
            train_a = train_a.permute(reshape)
            train_u = train_u.permute(reshape)
            test_a = test_a.permute(reshape)
            test_u = test_u.permute(reshape)
            
        print("Navier-Stokes (vis = 1e-3) Dataset has been loaded successfully!")
        print("X train shape:", train_a.shape, "Y train shape:", train_u.shape)
        print("X test shape:", test_a.shape, "Y test shape:", test_u.shape)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    elif type == "1e-4":
        ntrain = 8000
        ntest = 200
        total = ntrain + ntest
        f = h5py.File(path)
        data = f['u'][...,0:8200]
        data = torch.tensor(data,dtype=torch.float32)
        data = data.permute(3,1,2,0)

        # Traning data
        train_a = data[:ntrain,::sub,::sub,:T_in]
        train_u = data[:ntrain,::sub,::sub,T_in:T_out+T_in]
        # Testing data
        test_a = data[-ntest:,::sub,::sub,:T_in]
        test_u = data[-ntest:,::sub,::sub,T_in:T_out+T_in]
        
        if reshape:
            train_a = train_a.permute(reshape)
            train_u = train_u.permute(reshape)
            test_a = test_a.permute(reshape)
            test_u = test_u.permute(reshape)
            
        print("Navier-Stokes (vis = 1e-4) Dataset has been loaded successfully!")
        print("X train shape:", train_a.shape, "Y train shape:", train_u.shape)
        print("X test shape:", test_a.shape, "Y test shape:", test_u.shape)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    elif type == "1e-5":
        ntrain = 1100
        ntest = 100
        total = ntrain + ntest
        f = scipy.io.loadmat(path)
        data = f['u'][...,0:total]
        data = torch.tensor(data,dtype=torch.float32)
        print(data.shape)
        

        # Traning data
        train_a = data[:ntrain,::sub,::sub,:T_in]
        train_u = data[:ntrain,::sub,::sub,T_in:T_out+T_in]
        # Testing data
        test_a = data[-ntest:,::sub,::sub,:T_in]
        test_u = data[-ntest:,::sub,::sub,T_in:T_out+T_in]
        
        if reshape:
            train_a = train_a.permute(reshape)
            train_u = train_u.permute(reshape)
            test_a = test_a.permute(reshape)
            test_u = test_u.permute(reshape)
            
        print("Navier-Stokes (vis = 1e-5) Dataset has been loaded successfully!")
        print("X train shape:", train_a.shape, "Y train shape:", train_u.shape)
        print("X test shape:", test_a.shape, "Y test shape:", test_u.shape)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
        
    else:
        print("The type is unclaimed. Data loading is failed.")
        return
        
    return train_loader, test_loader

def navier_stokes_single(path, batch_size = 64, T_in = 10, T_out = 40, type = "1e-4", sub = 1,reshape = False):  
    f = scipy.io.loadmat(path)
    print(f["a"].shape)
    print(f["u"].shape)
    data = f["u"]
    loc = 0
    x = np.zeros([3980,256,256,1])
    y = np.zeros([3980,256,256,1])
    for i in range(20):
        for j in range(199):
            x[i,:,:,0:1] = data[i,:,:,j:j+1]
            y[i,:,:,0:1] = data[i,:,:,j+1:j+2]


    ntrain = 3600
    ntest = 200
    total = ntrain + ntest
    
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)

    if reshape:
        x = x.permute(reshape)
        y = y.permute(reshape)
    
    # Traning data
    x_train = x[:ntrain]
    y_train = y[:ntrain]
    # Testing data
    x_test = x[-ntest:]
    y_test = y[-ntest:]


    print("Navier-Stokes (vis = 1e-4) Dataset has been loaded successfully!")
    print("X train shape:", x_train.shape, "Y train shape:", y_train.shape)
    print("X test shape:", x_test.shape, "Y test shape:", y_test.shape)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def prepare_battery_data(batch_size=20, T_in=10, T_out=20, sub=1):
    # 准备训练和测试数据
    ntrain = 10
    ntest = 2
    total = ntrain + ntest

    # 从放电数据中提取电压和电流
    file_path = r'C:\Users\Crisy\Desktop\Matlab_Common_PredictionModel\B0038.mat'
    discharge_data = load_battery_data(file_path)
    voltage_data = discharge_data['24']['Voltage_load'][:total]
    current_data = discharge_data['24']['Current_load'][:total]

    # 创建输入数据 (电压和电流) 和输出数据 (预测未来的电压和电流)
    data = np.stack([voltage_data, current_data], axis=-1)  # 形状为 (样本数, 时间步长, 特征数)
    data = torch.tensor(data, dtype=torch.float32)
    data = data.permute(0, 2, 1)  # The dimension of the data shape is [Batch，特征, 时间步]

    # 划分训练和测试数据
    train_a = data[:ntrain, :, :T_in]
    train_u = data[:ntrain, :, T_in:T_in + T_out]
    test_a = data[-ntest:, :, :T_in]
    test_u = data[-ntest:, :, T_in:T_in + T_out]

    print("Battery Discharge Dataset has been loaded successfully!")
    print("X train shape:", train_a.shape, "Y train shape:", train_u.shape)
    print("X test shape:", test_a.shape, "Y test shape:", test_u.shape)
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    for data in train_loader:
        train_a_batch, train_u_batch = data
        print(train_a_batch.shape, train_u_batch.shape)
        break  # 只查看第一个 batch 的维度

    return train_loader, test_loader
