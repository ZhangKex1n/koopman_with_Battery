from koopmanlab.models import  kno
from koopmanlab import utils
from koopmanlab.models import koopmanViT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

class koopman:
    def __init__(self, backbone = "KNO1d", autoencoder = "MLP", o = 16, m = 16, r = 8, t_in = 1, device = False):
        self.backbone = backbone
        self.autoencoder = autoencoder
        self.operator_size = o
        self.modes = m
        self.decompose = r
        self.device = device
        self.t_in = t_in
        # Core Model
        self.params = 0
        self.kernel = False
        # Opt Setting
        self.optimizer = False
        self.scheduler = False
        self.loss_mse = torch.nn.MSELoss()
    def compile(self):
        if self.autoencoder == "MLP":
            encoder = kno.encoder_mlp(self.t_in, self.operator_size)
            decoder = kno.decoder_mlp(self.t_in, self.operator_size)
            print("The autoencoder type is MLP.")
        elif self.autoencoder == "MLP_norm":
            encoder = kno.encoder_mlp_norm(self.t_in, self.operator_size)
            decoder = kno.decoder_mlp_norm(self.t_in, self.operator_size)
            print("The autoencoder type is MLP.")
        elif self.autoencoder == "test":
            encoder = kno.encoder_test(input_dim =2 , hidden_dim = 64, latent_dim = 128)
            decoder = kno.decoder_test(latent_dim = 128, hidden_dim = 64, output_dim = 2)
        elif self.autoencoder == "Conv1d":
            encoder = kno.encoder_conv1d(self.t_in, self.operator_size)
            decoder = kno.decoder_conv1d(self.t_in, self.operator_size)
            print("The autoencoder type is Conv1d.")
        elif self.autoencoder == "Conv2d":
            encoder = kno.encoder_conv2d(self.t_in, self.operator_size)
            decoder = kno.decoder_conv2d(self.t_in, self.operator_size)
            print("The autoencoder type is Conv2d.")
        else:
            encoder = kno.encoder_mlp(self.t_in, self.operator_size)
            decoder = kno.decoder_mlp(self.t_in, self.operator_size)
            print("The autoencoder type is MLP.")
            print("Wrong!")
        if self.backbone == "KNO1d":
            self.kernel = kno.KNO1d(encoder, decoder, self.operator_size, modes_x = self.modes, decompose = self.decompose).to(self.device)
            print("KNO1d model is completed.")
        
        elif self.backbone == "KNO2d":
            self.kernel = kno.KNO2d(encoder, decoder, self.operator_size, modes_x = self.modes, modes_y = self.modes,decompose = self.decompose).to(self.device)
            print("KNO2d model is completed.")

        if not self.kernel == False:
            self.params = utils.count_params(self.kernel)
            print("Koopman Model has been compiled!")
            print("The Model Parameters Number is ",self.params)
    def opt_init(self, opt, lr, step_size, gamma):
        if opt == "Adam":
            self.optimizer = utils.Adam(self.kernel.parameters(), lr= lr, weight_decay=1e-4)
        if not step_size == False:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train_single(self, epochs, trainloader, evalloader = False):
        for ep in range(epochs):
            # Train
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                pred,im_re = self.kernel(xx)
                
                l_recons = self.loss_mse(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                l_pred = self.loss_mse(pred.reshape(bs, -1), yy.reshape(bs, -1))

                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()

                loss = 5*l_pred + 0.5*l_recons
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            mse_test = 0
            # Test
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        bs = xx.shape[0]
                        loss = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        pred,im_re = self.kernel(xx)


                        l_recons = self.loss_mse(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                        l_pred = self.loss_mse(pred.reshape(bs, -1), yy.reshape(bs, -1))


                        test_pred_full += l_pred.item()
                        test_recons_full += l_recons.item()
                        
                test_pred_full = test_pred_full/len(evalloader)
                test_recons_full = test_recons_full/len(evalloader)
                
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)

    def test_single(self, testloader):
        test_pred_full = 0
        test_recons_full = 0
        with torch.no_grad():
            for xx, yy in testloader:
                bs = xx.shape[0]
                loss = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)

                pred,im_re = self.kernel(xx)

                l_recons = self.loss_mse(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                l_pred = self.loss_mse(pred.reshape(bs, -1), yy.reshape(bs, -1))

                test_pred_full += l_pred.item()
                test_recons_full += l_recons.item()
        test_pred_full = test_pred_full/len(testloader)
        test_recons_full = test_recons_full/len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return test_pred_full


    def train(self, epochs, trainloader, step = 1, T_out = 30, evalloader = False):
        T_eval = T_out
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            ppp=0
            for xx, yy in trainloader:
                l_recons = 0
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                bs = xx.shape[0]
                for t in range(0, T_out):
                    y = yy[..., t:t + 1]

                    im,im_re = self.kernel(xx) #是kno里的x和x_reconstruct，im是经过decompose步koopman演化的值，im_re是仅经过编码器和解码器的输入
                    l_recons += self.loss_mse(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    if t == 0:
                        pred = im[...,-1:] #...表示保留前面所有的维度,意思是取im的
                    else:
                        pred = torch.cat((pred, im[...,-1:]), -1)
                    
                    xx = torch.cat((xx[..., step:], im[...,-1:]), dim=-1)

                w1= 0.99
                w2 =0.01
                weights = torch.tensor([w1, w2], device=xx.device)  # N为特征数量，可以根据实际情况设定权重
                l_pred = torch.mean(weights * self.loss_mse(pred, yy), dim=-1)  # 加权计算损失

                #l_pred = self.loss_mse(pred.reshape(bs, -1), yy.reshape(bs, -1))
                #print("l_pred & l_recons",l_pred,l_recons)
                loss = 100 * l_pred + 1.1 * l_recons
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()/T_out

                self.optimizer.zero_grad()
                loss.backward() #损失函数调用
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            #print("train_pred_full & train_recons_full & len(trainloader)",train_pred_full,train_recons_full,len(trainloader))
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            loc = 0
            mse_error = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        loss = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        for t in range(0, T_eval):
                            y = yy[..., t:t + 1]
                            im, im_re = self.kernel(xx)
                            l_recons += self.loss_mse(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            if t == 0:
                                pred = im[...,-1:]
                            else:
                                pred = torch.cat((pred, im[...,-1:]), -1)
                            xx = torch.cat((xx[..., 1:], im[...,-1:]), dim=-1)
                        l_pred = self.loss_mse(pred.reshape(bs, -1), yy.reshape(bs, -1))

                        test_recons_full += l_recons.item() / T_eval
                        test_pred_full += l_pred.item()
                        
                        loc = loc + 1
                    mse_error = mse_error / loc
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch |","Time","[Train Recons MSE]|","[Train Pred MSE]|","[Eval Recons MSE]|","[Eval Pred MSE]")
                print(ep,"|", t2 - t1,"|", train_recons_full,"|", train_pred_full,"|", test_recons_full,"|", test_pred_full)
            else:
                if ep == 0:
                    print("Epoch |","Time  |","Train Recons MSE|","Train Pred MSE|")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
    def test(self, testloader, step = 1, T_out = 30, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1]) #累计误差
        test_pred_full = 0 #累加测试集整体的预测误差
        test_recons_full = 0 #累加输入的重构误差
        loc = 0 #用于记录测试样本的批次数量
        with torch.no_grad():
            for xx, yy in testloader:
                loss = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[..., t:t + 1]
                    im, im_re = self.kernel(xx)
                    l_recons += self.loss_mse(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss_mse(im[..., -1:], y)
                    if t == 0:
                        pred = im[...,-1:]
                    else:
                        pred = torch.cat((pred, im[...,-1:]), -1)
                    time_error[t] = time_error[t] + t_error.item()
                    xx = torch.cat((xx[..., 1:], im[...,-1:]), dim=-1)

                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss_mse(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()
                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                
                #if(loc == 0 & is_plot):
                    # for i in range(T_out):
                    #     plt.subplot(1,3,1)
                    #     plt.title("Predict")
                    #     plt.imshow(pred[0,...,i].cpu().detach().numpy())
                    #     plt.subplot(1,3,2)
                    #     plt.imshow(yy[0,...,i].cpu().detach().numpy())
                    #     plt.title("Label")
                    #     plt.subplot(1,3,3)
                    #     plt.imshow(pred[0,...,i].cpu().detach().numpy()-yy[0,...,i].cpu().detach().numpy())
                    #     plt.title("Error")
                    #     plt.show()
                    #     plt.savefig(path + "time_"+str(i)+".png")
                    #     plt.close()
                if is_plot:
                    with torch.no_grad():
                        pred = pred.cpu().detach().numpy()
                        yy = yy.cpu().detach().numpy()
                        for i in range(pred.shape[0]):  # 遍历第一维度 (即 batch 中的样本数)
                            plt.figure(figsize=(10, 6))
                            plt.plot(range(1, pred.shape[2] + 1), pred[i, 0, :], label="Predicted Voltage", color='b')
                            plt.plot(range(1, yy.shape[2] + 1), yy[i, 0, :], label="Actual Voltage", color='r')
                            plt.xlabel("Time Step")
                            plt.ylabel("Voltage")
                            plt.title(f"Sample {i + 1}: Predicted vs Actual Voltage")
                            plt.legend()
                            plt.grid(True)
                            plt.tight_layout()
                            plt.show()
                            if is_save:
                                plt.savefig(path + f"sample_{i}.png")
                            plt.close()
                loc = loc + 1
        test_pred_full = test_pred_full / (loc+1)
        test_recons_full = test_recons_full / (loc+1)
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return time_error
        
    def save(self, path):
        (fpath,_) = os.path.split(path)
        if not os.path.isfile(fpath):
            os.makedirs(fpath)
        torch.save({"koopman":self,"model":self.kernel,"model_params":self.kernel.state_dict()}, path)

class koopman_vit:
    def __init__(self, decoder = "Conv2d", depth = 16, resolution=(256, 256), patch_size=(4, 4),
            in_chans=1, out_chans=1, embed_dim=768, parallel = False, device = False):
        # Model Hyper-parameters
        self.decoder = decoder
        self.resolution = resolution
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.depth = depth
        # Core Model
        self.params = 0
        self.kernel = False
        # Opt Setting
        self.optimizer = False
        self.scheduler = False
        self.device = device
        self.parallel = parallel
        self.loss = torch.nn.MSELoss()
    def compile(self):
        self.kernel = koopmanViT.ViT(img_size=self.resolution, patch_size=self.patch_size, in_chans=self.in_chans, out_chans=self.out_chans, num_blocks=self.num_blocks, embed_dim = self.embed_dim, depth=self.depth, settings = self.decoder).to(self.device)
        if self.parallel:
            self.kernel = torch.nn.DataParallel(self.kernel)
        self.params = utils.count_params(self.kernel)
        
        print("Koopman Fourier Vision Transformer has been compiled!")
        print("The Model Parameters Number is ",self.params)
        
    def opt_init(self, opt, lr, step_size, gamma):
        if opt == "Adam":
            self.optimizer = utils.Adam(self.kernel.parameters(), lr= lr, weight_decay=1e-4)
        if not step_size == False:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
    def train_multi(self, epochs, trainloader, T_out = 10, evalloader = False):
        T_eval = T_out
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for xx, yy in trainloader:
                l_recons = 0
                xx = xx.to(self.device) # [batchsize,1,x,y]
                yy = yy.to(self.device) # [batchsize,T,x,y]
                bs = xx.shape[0]
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im,im_re = self.kernel(xx)
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    
                    if t == 0:
                        pred = im[:, -1:]
                    else:
                        pred = torch.cat((pred, im[:, -1:]), -1)
                    
                    xx = im
                
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                loss = 5 * l_pred + 0.5 * l_recons
                
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()/T_out

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            loc = 0
            mse_error = 0
            if evalloader:
                with torch.no_grad():
                    for xx, yy in evalloader:
                        loss = 0
                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        for t in range(0, T_eval):
                            y = yy[:, t:t + 1]
                            im, im_re = self.kernel(xx)
                            
                            l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                            
                            if t == 0:
                                pred = im
                            else:
                                pred = torch.cat((pred, im), 1)
                                
                            xx = im
                            
                        l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))

                        test_recons_full += l_recons.item() / T_eval
                        test_pred_full += l_pred.item()
                        
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
    
    def test_multi(self, testloader, step = 1, T_out = 5, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        with torch.no_grad():
            for xx, yy in testloader:
                loss = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im, im_re = self.kernel(xx)
                    
                    
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im, y)
                    
                    xx = im
                    
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    time_error[t] = time_error[t] + t_error.item()
    
                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()

                # 训练完成后绘制结果
                if is_plot:
                    with torch.no_grad():
                        pred = pred[:, 0, :].cpu().detach().numpy()  # 只取电压维度
                        yy = yy[:, 0, :].cpu().detach().numpy()  # 只取电压维度
                        for i in range(T_out):
                            plt.figure(figsize=(12, 4))

                            # 绘制预测和标签的折线图
                            plt.subplot(1, 2, 1)
                            plt.title("Predict vs Label")
                            plt.plot(pred[0, :], label="Predicted Voltage")
                            plt.plot(yy[0, :], label="Actual Voltage")
                            plt.legend()

                            # 绘制误差
                            plt.subplot(1, 2, 2)
                            plt.title("Error")
                            plt.plot(pred[0, :] - yy[0, :], label="Error")
                            plt.legend()

                            plt.tight_layout()
                            plt.show()
                            if is_save:
                                plt.savefig(path + f"time_{i}.png")
                            plt.close()
                loc = loc + 1
        test_pred_full = test_pred_full / loc
        test_recons_full = test_recons_full / loc
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        return time_error
        
        
    def train_single(self, epochs, trainloader, evalloader = False):
        for ep in range(epochs):
            self.kernel.train()
            t1 = default_timer()
            train_recons_full = 0
            train_pred_full = 0
            for x, y in trainloader:
                l_recons = 0
                x = x.to(self.device) # [batchsize,1,64,64]
                y = y.to(self.device) # [batchsize,1,64,64]
                bs = x.shape[0]
                
                im,im_re = self.kernel(x)
                
                l_recons = self.loss(im_re.reshape(bs, -1), x.reshape(bs, -1))
                l_pred = self.loss(im.reshape(bs, -1), y.reshape(bs, -1))
                
                loss = 5 * l_pred + 0.5 * l_recons
                
                train_pred_full += l_pred.item()
                train_recons_full += l_recons.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_pred_full = train_pred_full / len(trainloader)
            train_recons_full = train_recons_full / len(trainloader)
            t2 = default_timer()
            test_pred_full = 0
            test_recons_full = 0
            loc = 0
            mse_error = 0
            if evalloader:
                with torch.no_grad():
                    for x, y in evalloader:
                        loss = 0
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        im, im_re = self.kernel(x)

                        l_recons = self.loss(im_re.reshape(bs, -1), x.reshape(bs, -1))
                        l_pred = self.loss(im.reshape(bs, -1), y.reshape(bs, -1))

                        test_recons_full += l_recons.item()
                        test_pred_full += l_pred.item()
                        
                test_recons_full = test_recons_full / len(evalloader)
                test_pred_full = test_pred_full / len(evalloader)
            self.scheduler.step()

            if evalloader:
                if ep == 0:
                    print("Epoch","Time","[Train Recons MSE]","[Train Pred MSE]","[Eval Recons MSE]","[Eval Pred MSE]")
                print(ep, t2 - t1, train_recons_full, train_pred_full, test_recons_full, test_pred_full)
            else:
                if ep == 0:
                    print("Epoch","Time","Train Recons MSE","Train Pred MSE")
                print(ep, t2 - t1, train_recons_full, train_pred_full)
                
    def test_single(self, testloader, T_out = 1, path = False, is_save = False, is_plot = False):
        time_error = torch.zeros([T_out,1])
        test_pred_full = 0
        test_recons_full = 0
        loc = 0
        idx = np.random.randint(0,len(testloader))
        with torch.no_grad():
            for xx, yy in testloader:
                loss = 0
                bs = xx.shape[0]
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                l_recons = 0
                for t in range(0, T_out):
                    y = yy[:, t:t + 1]
                    im, im_re = self.kernel(xx)
                    
                    l_recons += self.loss(im_re.reshape(bs, -1), xx.reshape(bs, -1))
                    t_error = self.loss(im, y)
                    
                    xx = im
                    
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    time_error[t] = time_error[t] + t_error.item()
    
                test_recons_full += l_recons.item() / T_out
                l_pred = self.loss(pred.reshape(bs, -1), yy.reshape(bs, -1))
                test_pred_full += l_pred.item()

                if(loc == 0 & is_save):
                    torch.save({"pred":pred, "yy":yy}, path+ "pred_yy.pt")
                
                if(loc == 0 & is_plot):
                    for i in range(T_out):
                        plt.subplot(1,3,1)
                        plt.title("Predict")
                        plt.imshow(pred[0,i].cpu().detach().numpy())
                        plt.subplot(1,3,2)
                        plt.imshow(yy[0,i].cpu().detach().numpy())
                        plt.title("Label")
                        plt.subplot(1,3,3)
                        plt.imshow(pred[0,i].cpu().detach().numpy()-yy[0,i].cpu().detach().numpy())
                        plt.title("Error")
                        plt.show()
                        plt.savefig(path + "time_"+str(i)+".png")
                        plt.close()
                loc = loc + 1

        test_pred_full = test_pred_full / len(testloader)
        test_recons_full = test_recons_full / len(testloader)
        time_error = time_error / len(testloader)
        print("Total prediction test mse error is ",test_pred_full)
        print("Total reconstruction test mse error is ",test_recons_full)
        
        return time_error
        
    def save(self, path):
#        (fpath,_) = os.path.split(path)
#        print(fpath, os.path.isfile(fpath))
#        if not os.path.isfile(fpath):
#            os.makedirs(fpath)
        torch.save({"koopman":self,"model":self.kernel,"model_params":self.kernel.state_dict()}, path)
