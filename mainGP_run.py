import math
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import gpytorch
from matplotlib import pyplot as plt
from pyKriging.samplingplan import samplingplan
import pandas as pd
from time import time
from scipy.interpolate import griddata
import os
import numpy as np
import random
from GPy import *
from opt import *
import numpy as np

path1=r".\Database\saveX_gpytorch_multi_EI_MS_H.npy"
path2=r".\Database\savey_gpytorch_multi_EI_MS_H.npy"
path1n=r".\ROM\E3\saveX_gpytorch_multi_EI_MS %d.npy" %744
path2n=r".\ROM\E3\savey_gpytorch_multi_EI_MS %d.npy" %744
path3=r".\Database\saveTestXdict_gpytorch_multi_EI_MS.npy"

path5='.\Database\singlefidelity_high_database.pth'
#path5='.\Database\singlefidelity_low_database.pth'
UPBound = np.array(UPB).T
LOWBound = np.array(LOWB).T

pathx=r'.\Database\train_x.csv'
pathy=r'.\Database\train_y.csv'


init_sample=8*8
training_iter=30#55#110
Infillpoints=8*2
Episode=87#47
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
num_tasks=2
Offline=0
dict = [i for i in range(TestX.shape[0])]
testmode="experiment"#DTLZ#WFG

dim = np.array(UPB).shape[0]
def save():

    np.save(path1, np.array(train_x.cpu()))
    np.savetxt(r'.\Database\train_x.csv', np.array(train_x.cpu()), delimiter=',')
    np.savetxt(r'.\Database\train_y.csv', np.array(train_y.cpu()), delimiter=',')
    np.save(path2, np.array(train_y.cpu()))
    np.save(path3, np.array(dict))

    np.save(r".\ROM\E3\saveX_gpytorch_multi_EI_MS %d.npy" % (len(train_y)), np.array(train_x.cpu()))
    np.save(r".\ROM\E3\savey_gpytorch_multi_EI_MS %d.npy" % (len(train_y)), np.array(train_y.cpu()))
    np.save(r".\ROM\E3\saveTestXdict_gpytorch_multi_EI_MS %d.npy" % (len(train_y)), np.array(dict))
    torch.save(model.state_dict(), path5)
def MAE(model, likelihood):
    # 使用numpy的random.uniform函数，生成一个80行7列的随机矩阵，每个元素在对应的区间内
    #X = np.random.uniform(LOWBound, UPBound, (80, 7))
    X = train_x[-80:,:]  # 80x7 matrix
    noise = np.random.uniform(-0.01, 0.01, X.shape)  # generate noise with the same shape as X
    #X_noisy = X + noise  # add noise to X
    X[:,0:2]=X[:,0:2]-0.01
    test_x = torch.tensor(X).to(device)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_y2 = likelihood(
            *model(test_x.to(torch.float32), test_x.to(torch.float32)))
        observed_pred_y21 = observed_pred_y2[0].mean
        observed_pred_y22 = observed_pred_y2[1].mean
        v21=observed_pred_y2[0].variance
        v22 = observed_pred_y2[1].variance
    # test_y_actual1 = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
    print("测试点预测值(推力：效率)", observed_pred_y21, observed_pred_y22 )
    print("测试点variance(推力：效率)", v21, v22 )
    x,test_y_actual = findpointOL(test_x, num_tasks, mode=testmode)
    print("测试点真实值(推力：效率)", test_y_actual)
    print("数据集(推力：效率)", train_y[-80:, :])
    delta_y11 = torch.abs(observed_pred_y21 - test_y_actual[:, 0]).detach().numpy()
    delta_y12 = torch.abs(observed_pred_y22 - test_y_actual[:, 1]).detach().numpy()
    print("MAE测试平均误差", np.mean(delta_y11), np.mean(delta_y12))
    return np.mean(delta_y11), np.mean(delta_y12)
if __name__=="__main__":
    for j in range(8):
        # Construct the folder name
        folder_name =  f'MMGP_OL{j}'
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    if os.path.exists(path1):
        initialDataX=np.load(path1,allow_pickle=True)
        initialDataY=np.load(path2,allow_pickle=True)

        #initialDataX=np.loadtxt(pathx, delimiter=',')
        #initialDataY=np.loadtxt(pathy,delimiter=',')


        dict=np.load(path3,allow_pickle=True).astype(int).tolist()
        #自检程序
        mask = np.any(np.isnan(initialDataY) | np.isinf(initialDataY), axis=1)
        # 根据掩码删除对应的行
        cleaned_initialDataY = initialDataY[~mask]
        cleaned_initialDataX = initialDataX[~mask]
        # 更新 dict 列表，删除对应行的索引
        #cleaned_dict = [item for i, item in enumerate(dict) if not mask[i]]
        # 更新变量的值
        initialDataY = cleaned_initialDataY
        initialDataX = cleaned_initialDataX
        #dict = cleaned_dict
    else:
        sp = samplingplan(dim)
        X = sp.optimallhc(init_sample)
        if testmode=="experiment":
            X= LOWBound+X*(UPBound-LOWBound)
        else:
            pass

        initialDataX=np.zeros([init_sample,dim])
        initialDataY = np.zeros([init_sample,np.abs(num_tasks)])
        if Offline == 1:
            for index, value in enumerate(X):
                initialDataX[index, :], initialDataY[index:index + 1, :] = findpoint_interpolate(value, Frame, 2)
                if np.isnan(initialDataY[index, 0]) or np.isnan(initialDataY[index, 1]):
                    initialDataX[index, :], initialDataY[index, :] = findpoint_interpolate(value, Frame, 2, "nearest")
        else:
            ##online
            initialDataX, initialDataY = findpointOL(X,num_task=2,mode=testmode)
        initialDataX = normalizer.normalize(X)


    train_x=torch.tensor(initialDataX).to(device).to(torch.float32)
    train_y=torch.tensor(initialDataY).to(device).to(torch.float32)
    #save()
    # independent Multitask
    likelihood1 = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(device)
    model1 = SpectralMixtureGPModelBack(train_x,train_y[:,0], likelihood1).to(device)
    #model1 =ExactGPModel(train_x,train_y[:,0], likelihood1).to(device)
    likelihood2 = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(device)
    model2 = SpectralMixtureGPModelBack(train_x,train_y[:,1], likelihood2).to(device)
    #model2 = ExactGPModel(train_x, train_y[:, 1], likelihood2).to(device)
    model = gpytorch.models.IndependentModelList(model1, model2).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)

    from gpytorch.mlls import SumMarginalLogLikelihood
    mll = SumMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    cofactor = [0.5, 0.5]
    for j in range(Episode):
        print("Episode",j,"point",len(train_y))
        model.train()
        likelihood.train()
        if os.path.exists(path5):
            state_dict = torch.load(path5)
            model.load_state_dict(state_dict)
        with gpytorch.settings.cholesky_jitter(1e-0):
            for i in range(training_iter):
                optimizer.zero_grad()
                output = model(*model.train_inputs)
                loss = -mll(output, model.train_targets)
                loss.backward(retain_graph=True)
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
                optimizer.step()
        #save()
        #################test###################
        IGD,pop=optIGD(model, likelihood, num_task=num_tasks, testmode=testmode, train_x=train_x)
        #with gpytorch.settings.cholesky_jitter(1e-0):
        #    M=MAE(model,likelihood)
        #f = open("./MAE.txt", "a", encoding="utf - 8")
        #f.writelines((str(train_y.shape[0]) + ",", str( M[0]) + ",", str( M[1])+ ",", str(IGD) + "\n"))
        #f.close()
        ##########################################################infill###################
        X,Y=infillGA(model, likelihood, Infillpoints, dict, num_tasks,"EI", device=device, cofactor=cofactor, y_max=[torch.max(train_y[:,0]).item() ,torch.max(train_y[:,1]).item()], offline=Offline,train_x=train_x,testmode=testmode,final_population_X=pop,norm=normalizer)
        cofactor=UpdateCofactor(model,likelihood,X.to(torch.float32),Y.to(torch.float32),cofactor,torch.max(train_y,dim=0).values-torch.min(train_y,dim=0).values)
        #cofactor=[0.5,0.5]
        print("addpoint",X)
        train_x=torch.cat((train_x,X),dim=0).to(torch.float32)
        train_y=torch.cat((train_y,Y),dim=0).to(torch.float32)
        model1 = SpectralMixtureGPModelBack(train_x, train_y[:, 0], likelihood1).to(device)
        likelihood2 = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(device)
        model2 = SpectralMixtureGPModelBack(train_x, train_y[:, 1], likelihood2).to(device)
        model = gpytorch.models.IndependentModelList(model1, model2).to(device)
        model.train()
        likelihood.train()
        save()








