import math
import numpy as np
import torch
import gpytorch
from pyKriging.samplingplan import samplingplan
import time
import os
from GPy import *
from opt import *

from gpytorch import kernels, means, models, mlls, settings, likelihoods, constraints, priors
from gpytorch import distributions as distr
path5='.\Database\multifidelity_database.pth'

path1csv=r".\Database\saveX_gpytorch_multifidelity_multitask.csv"
path2csv=r".\Database\saveI_gpytorch_multifidelity_multitask.csv"
path3csv=r".\Database\saveY_gpytorch_multifidelity_multitask.csv"
path4csv=r".\Database\saveTestXdict_gpytorch_multifidelity_multitask.csv"
#lowfidelity
#pathx2=r".\ROM\E3\saveX_gpytorch_multi_EI_MS 928.npy"
#pathy2=r".\ROM\E3\savey_gpytorch_multi_EI_MS 928.npy"
pathx2=r".\Database\saveX_gpytorch_multi_EI_MS.npy"
pathy2=r".\Database\savey_gpytorch_multi_EI_MS.npy"

path1H=r".\Database\saveX_gpytorch_multi_EI_MS_H.npy"
path2H=r".\Database\savey_gpytorch_multi_EI_MS_H.npy"

# We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
# n = 250
init_sample = 8*5
Infillpoints=8*2
training_iterations = 13#33#5
num_tasks=-2
num_input=len(UPB)
Episode=3
LowSample=1800
testsample=140
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")
Offline=0
testmode="experiment"
UPBound = np.array(UPB).T
LOWBound = np.array(LOWB).T
dict = [i for i in range(TestX.shape[0])]
def save():
    np.savetxt(path1csv, np.array(full_train_x.cpu()),delimiter=',')
    np.savetxt(path2csv, np.array(full_train_i.cpu()),delimiter=',')
    np.savetxt(path3csv, np.array(full_train_y.cpu()),delimiter=',')

    np.savetxt(r".\ROM\E2\saveX_gpytorch_multifidelity_multitask %d.csv" % (torch.sum(full_train_i).item()),
            np.array(full_train_x.cpu()),delimiter=',')
    np.savetxt(r".\ROM\E2\saveI_gpytorch_multifidelity_multitask %d.csv" % (torch.sum(full_train_i).item()),
            np.array(full_train_i.cpu()),delimiter=',')
    np.savetxt(r".\ROM\E2\saveY_gpytorch_multifidelity_multitask %d.csv" % (torch.sum(full_train_i).item()),
            np.array(full_train_y.cpu()),delimiter=',')

    np.savetxt(r'.\Database\train_x.csv', np.array(full_train_x.cpu()), delimiter=',')
if os.path.exists(path1csv):
    full_train_x=torch.FloatTensor(np.loadtxt(path1csv, delimiter=','))
    full_train_i=torch.FloatTensor(np.loadtxt(path2csv,delimiter=','))
    full_train_y=torch.FloatTensor(np.loadtxt(path3csv, delimiter=','))
    full_train_i=torch.unsqueeze(full_train_i,dim=1)
    if os.path.exists(path4csv):
        dict = np.loadtxt(path4csv,  delimiter=',').astype(int).tolist()
else:
    #train_x1 = torch.tensor(np.load(pathx2, allow_pickle=True)).to(torch.float32)
    #train_y1 =torch.tensor( np.load(pathy2, allow_pickle=True)).to(torch.float32)
    train_x1 = torch.tensor(np.loadtxt(r'.\Database\train_x0015.csv',  delimiter=',')).to(torch.float32)
    train_y1 =torch.tensor( np.loadtxt(r'.\Database\train_y0015.csv',  delimiter=',')).to(torch.float32)

    # High fidelity
    sp = samplingplan(num_input)
    X = sp.optimallhc(init_sample)
    initialDataX = normalizer.denormalize(X)
    #X= LOWBound+X*(UPBound-LOWBound)
    train_x2=np.zeros([init_sample, num_input])
    train_y2 = np.zeros([init_sample,abs(num_tasks)])
    size=len(Frame.iloc[:, 4].to_numpy())



    if Offline == 1:
        for index, value in enumerate(X):
            train_x2[index, :], train_y2[index, :] = findpoint_interpolate(value, Frame, num_tasks=num_tasks)
            if np.isnan(train_y2[index, 0]) or np.isnan(train_y2[index, 1]):
                train_x2[index, :], train_y2[index, :] = findpoint_interpolate(value, Frame, num_tasks=num_tasks,
                                                                               method="nearest")
    elif Offline == 0:
        ##online
        print(initialDataX)
        train_x2, train_y2 = findpointOL(initialDataX,num_task=num_tasks)
    else :
        train_x2 = torch.tensor(np.load(path1H, allow_pickle=True)).to(torch.float32)
        train_y2 = torch.tensor(np.load(path2H, allow_pickle=True)).to(torch.float32)

    X = torch.tensor(X).to(device).to(torch.float32)
    train_y2 = torch.tensor(train_y2).to(device).to(torch.float32)

    # Construct data

    train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0) #low
    train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1) #high

    full_train_x = torch.cat([train_x1, X]).to(device)
    full_train_i = torch.cat([train_i_task1, train_i_task2]).to(device)
    full_train_y = torch.cat([train_y1, train_y2]).to(device)
    # Construct data2
    np.savetxt(path1csv, np.array(full_train_x.cpu()),delimiter=',')
    np.savetxt(path2csv, np.array(full_train_i.cpu()),delimiter=',')
    np.savetxt(path3csv, np.array(full_train_y.cpu()),delimiter=',')

# Here we have two iterms that we're passing in as train_inputs
likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
#50: 0:2565
model1 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,0], likelihood1).to(device)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model2 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,1], likelihood2).to(device)
model = gpytorch.models.IndependentModelList(model1, model2).to(device)
likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)

print(model)

# #----------------------------------------- cuda
# full_train_x = full_train_x.cuda()
# full_train_i = full_train_i.cuda()
# full_train_y = full_train_y.cuda()

# model = model.cuda()
# likelihood = likelihood.cuda()
# #----------------------------------------- cuda
# "Loss" for GPs - the marginal log likelihood
from gpytorch.mlls import SumMarginalLogLikelihood
mll = SumMarginalLogLikelihood(likelihood, model)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

cofactor = [0.5, 0.5]
for i in range(Episode):
    print(  "Episode%d-point %d : %d "%(i, torch.sum(full_train_i).item(),len(full_train_i)-torch.sum(full_train_i).item())   )
    if os.path.exists(path5):
         state_dict = torch.load(path5)
         model.load_state_dict(state_dict)
    # else:
    for j in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward(retain_graph=True)
        print('Iter %d/%d - Loss: %.3f' % (j + 1,training_iterations, loss.item()))
        optimizer.step()
    torch.save(model.state_dict(), path5)
    #save()
    IGD, pop = optIGD(model, likelihood, num_task=num_tasks, testmode=testmode, train_x=full_train_x)
    X, Y = infillGA(model, likelihood, Infillpoints, dict, num_tasks, "EI", device=device, cofactor=cofactor,
                y_max=[torch.max(full_train_y[:int(torch.sum(full_train_i).item()), 0]).item(), torch.max(full_train_y[:int(torch.sum(full_train_i).item()), 1]).item()], offline=Offline,
                train_x=full_train_x,testmode=testmode,final_population_X=pop,norm=normalizer)
    full_train_x = torch.cat((full_train_x, X), dim=0).to(torch.float32).to(device)
    full_train_i = torch.cat((full_train_i, torch.ones(Infillpoints).unsqueeze(-1)), dim=0).to(torch.float32).to(device)
    full_train_y = torch.cat((full_train_y, Y), dim=0).to(torch.float32).to(device)
    save()
    model1 = MultiFidelityGPModel((full_train_x, full_train_i), full_train_y[:, 0], likelihood1).to(device)
    model2 = MultiFidelityGPModel((full_train_x, full_train_i), full_train_y[:, 1], likelihood2).to(device)
    model = gpytorch.models.IndependentModelList(model1, model2).to(device)

    model.train()
    likelihood.train()


    #save()


    cofactor,MAE = UpdateCofactor(model, likelihood, X.to(torch.float32), Y.to(torch.float32), cofactor,
                          torch.max(full_train_y[:int(torch.sum(full_train_i).item()), :], dim=0).values-torch.min(full_train_y[:int(torch.sum(full_train_i).item()), :], dim=0).values
                                  ,MFkernel=1)
    print("addpoint", X, "MAE", MAE)


#TEST测试
X=full_train_x[0:50,:]
test_x=torch.tensor(X).to(device)
test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y2 = likelihood(*model((test_x.to(torch.float32), test_i_task2), (test_x.to(torch.float32), test_i_task2)))
    observed_pred_y21 = observed_pred_y2[0].mean
    observed_pred_y22 = observed_pred_y2[1].mean
# test_y_actual1 = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
print("测试点预测值(推力：效率)",observed_pred_y21,observed_pred_y22*0.1)
#test_x, test_y_actual = findpointOL(X, num_task=num_tasks)
test_y_actual=full_train_y[0:50,:]
print("测试点真实值(推力：效率)", test_y_actual)
delta_y11 = torch.abs(observed_pred_y21 - test_y_actual[:,0]).detach().numpy()
delta_y12 = torch.abs(observed_pred_y22 - test_y_actual[:,1]).detach().numpy()
print("MAE测试平均误差",np.mean(delta_y11),np.mean(delta_y12))
#print("MAE测试平均误差百分比", np.mean(delta_y11),"% 和", np.mean(delta_y12)*100/2*0.1,"%")


