import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# The structure of Auto-Encoder

class encoder_test(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(encoder_test, self).__init__()
        #输入数据维度[5, 2, 70]-->(batch_size, input_dim, sequence_length)
        # self.network = nn.Sequential(
        #     nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),  # 使用 1D 卷积来处理时间序列数据
        #     nn.LeakyReLU(),  # 使用Leaky ReLU作为激活函数
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        #     nn.LeakyReLU(),  # 使用Leaky ReLU作为激活函数
        #     nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1),
        #输出维度为 (batch_size, latent_dim = 20, sequence_length = 70)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # 用于将LSTM的输出映射到潜在空间
        self.fc = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        # x: [batch_size, input_dim, seq_len] 转置为 [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 对每个时间步的隐藏状态映射到潜在空间，得到 [batch_size, seq_len, latent_dim]
        latent_out = self.fc(lstm_out)
        # 转置回 [batch_size, latent_dim, seq_len]
        latent_out = latent_out.permute(0, 2, 1)

        return latent_out  # 输出维度: [batch_size, latent_dim, seq_len]

        #x:[5,20,70]

class decoder_test(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(decoder_test, self).__init__()
        # self.network = nn.Sequential(
        #     #nn.Linear(latent_dim, hidden_dim),  # 潜在空间到隐藏层的线性变换
        #     nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=3, padding=1),  # 反卷积层
        #     nn.LeakyReLU(),  # 使用Leaky ReLU作为激活函数
        #     #nn.Linear(hidden_dim, hidden_dim),  # 隐藏层之间的线性变换
        #     nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # 反卷积层
        #     nn.LeakyReLU(),  # 使用Leaky ReLU作为激活函数
        #     #nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层的线性变换
        #     nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        # )
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out)  # 解码器将LSTM输出映射回原始维度
        output = output.permute(0, 2, 1)
        return output

class encoder_mlp(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_mlp, self).__init__()
        self.layer = nn.Linear(t_len, op_size)
    def forward(self, x):
        x = self.layer(x)
        return x

class decoder_mlp(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_mlp, self).__init__()
        self.layer = nn.Linear(op_size, t_len)
    def forward(self, x):
        x = self.layer(x)
        return x

class encoder_conv1d(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(t_len, op_size,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class decoder_conv1d(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(op_size, t_len,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class encoder_conv2d(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(t_len, op_size,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x

class decoder_conv2d(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(op_size, t_len,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x


# Koopman 1D structure
class Koopman_Operator1D(nn.Module):
    def __init__(self, op_size, modes_x = 16,t_len = 70):
        super(Koopman_Operator1D, self).__init__()
        self.op_size = op_size
        self.scale = (1 / (op_size * op_size))
        self.modes_x = modes_x
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(t_len, op_size, 16, dtype=torch.cfloat))
        #
        self.print_count = 0
    # Complex multiplication
    def time_marching(self, input, weights):
        # 输出维度为 (batch_size, latent_dim = 20, sequence_length = 70)
        # input:(batch, t, x), weights(koopman算子):(t, t+1, x) -> (batch, t+1, x)
        # f (output features or new states): 输出特征或新状态变量的数 应该是20
        # b (batch size): 批量大小，用于表示数据中每个样本的数量 应该是5
        # t (time steps): 时间步长，通常用于表示时间序列数据中的步数 应该是70
        # x (features or state variables): 特征或状态变量的数量，用于表示每个时间步中包含多少特征（例如电压、电流等传感器数据 应该是16
        return torch.einsum("btx,tfx->bfx", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform

        x_ft = torch.fft.rfft(x) #傅里叶变换，x_ft是输入x的频域表示
        #x_ft[5,70,65]
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        #out_ft维度和x_ft一样
        #out_ft是一个初始化的与x_ft形状相同的零张量，用于存放time_marching(koopman算子推进)的频域数据
        out_ft[:, :, :self.modes_x] = self.time_marching(x_ft[:, :, :self.modes_x], self.koopman_matrix)

        print("aa:",self.print_count)
        self.print_count += 1
        #只有频率中的低频模式被用来推进时间
        #Inverse Fourier Transform
        x = torch.fft.irfft(out_ft, n=x.size(-1)) #傅里叶逆变换
        return x

class KNO1d(nn.Module):
    def __init__(self, encoder, decoder, op_size, modes_x = 16, decompose = 4, linear_type = True, normalization = False):
        super(KNO1d, self).__init__()
        # Parameter
        self.op_size = op_size
        self.decompose = decompose
        # Layer Structure
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = Koopman_Operator1D(self.op_size, modes_x = modes_x)
        #self.w0 = nn.Conv1d(op_size, op_size, 1) #高频补充量
        self.lstm = nn.LSTM(input_size=op_size, hidden_size=op_size, num_layers=1, batch_first=True)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(op_size)
    def forward(self, x):
        # Reconstruct
        x_reconstruct = self.enc(x)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 2, 1)
        x_w = x

        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + x1 #x是input，x1是增量
            else:
                x = torch.tanh(x + x1)

        #x_lstm, _ = self.lstm(x)

        if self.normalization:
            x = torch.tanh(self.norm_layer(x)) #将高频补充量和koopman演化量加在一起
        else:
            x = torch.tanh(x)
            #x = torch.tanh(x_lstm + x)
        x = x.permute(0, 2, 1)
        x = self.dec(x) # Decoder
        return x, x_reconstruct

# Koopman 2D structure
class Koopman_Operator2D(nn.Module):
    def __init__(self, op_size, modes_x, modes_y):
        super(Koopman_Operator2D, self).__init__()
        self.op_size = op_size
        self.scale = (1 / (op_size * op_size))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x,y ), (t, t+1, x,y) -> (batch, t+1, x,y)
        return torch.einsum("btxy,tfxy->bfxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft2(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class KNO2d(nn.Module):
    def __init__(self, encoder, decoder, op_size, modes_x = 10, modes_y = 10, decompose = 6, linear_type = True, normalization = False):
        super(KNO2d, self).__init__()
        # Parameter
        self.op_size = op_size
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        # Layer Structure
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = Koopman_Operator2D(self.op_size, self.modes_x, self.modes_y)
        self.w0 = nn.Conv2d(op_size, op_size, 1)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(op_size)
    def forward(self, x):
        # Reconstruct
        x_reconstruct = self.enc(x)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 3, 1, 2)
        x_w = x
        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
        x = x.permute(0, 2, 3, 1)
        x = self.dec(x) # Decoder
        return x, x_reconstruct
