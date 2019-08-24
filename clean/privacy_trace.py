#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from matplotlib.dates import datestr2num
import torch.optim as optim
from sklearn import preprocessing
import math
from scipy.stats import gaussian_kde
from datetime import datetime
import random

torch.cuda.init()
CUDA1 = torch.device('cuda:1')

class ChaniaDataset(Dataset):
    def __init__(self, csv_file, num_features, transform=None, normalize=True):
        self.num_features = num_features
        self.augmented_data = pd.read_csv(csv_file, usecols = list(range(1,num_features+1)))
        self.userlabels = pd.read_csv(csv_file, usecols=[0])
        self.transform = transform

        if normalize:
            self.augmented_data=(self.augmented_data-self.augmented_data.mean())/self.augmented_data.std()

        self.tensor_data = torch.from_numpy(self.augmented_data.values).cuda(CUDA1)        
        self.tensor_labels = torch.from_numpy(self.userlabels.values).cuda(CUDA1)

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        # data = self.augmented_data.iloc[idx].values
        # data = data.astype('float').reshape(-1,self.num_features)
        data = self.tensor_data[idx]
        data = data.view(-1,self.num_features)
        # user = self.userlabels.iloc[idx].values
        # user = user.astype('int').reshape(-1,1)
        user = self.tensor_labels[idx]
        user = user.view(-1,1).long()
        sample = {'x':data, 'u':user}
        # if self.transform:
            # sample = self.transform(sample)
        return sample

# class ToTensor(object):
#     def __call__(self, sample):
#         data, user = sample['x'], sample['u']
#         return {'x':torch.from_numpy(data), 'u':torch.from_numpy(user)}

## HELPER FUNCTIONS

def poly(degree, long, lat):
    return torch.cat([long**i*lat**(degree-i) for degree in range(degree+1) for i in range(degree,-1,-1)],1)

def signal_map_params(x,degree, n):
    polynomial = poly(degree, x[:,:,[2+3*i for i in range(n)]].reshape(-1,1),x[:,:,[1+3*i for i in range(n)]].reshape(-1,1))
    beta = torch.mm(torch.inverse(torch.mm(torch.transpose(polynomial,0,1), polynomial)),
                  torch.mm(torch.transpose(polynomial,0,1), x[:,:,[0+3*i for i in range(n)]].reshape(-1,1)))
    return beta

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.008)
        # torch.nn.init.normal_(m.bias,mean=0.25, std=0.1)
        # m.bias.data.fill_(2)

def analytical_gaussian_sigma(eta, epsilon, delta):
    def phi(x):
        return 1/2*(1+math.erf(x/math.sqrt(2)))
    delta_0 = phi(0)-math.e**epsilon*phi(-math.sqrt(2*epsilon))
    def Bplus(v):
        return phi(math.sqrt(epsilon*v))-math.e**epsilon*phi(-math.sqrt(epsilon*(v+2)))
    def Bminus(v):
        return phi(-math.sqrt(epsilon*v))-math.e**epsilon*phi(-math.sqrt(epsilon*(v+2)))
    if delta >= delta_0:
        vstar = 0
        while Bplus(vstar) <= delta:
            vstar += 1
            if vstar == 1000000:
                return 0
        alpha = math.sqrt(1+vstar/2)-math.sqrt(vstar/2)
    else:
        ustar = 0
        while Bminus(ustar) > delta:
            ustar += 1
        alpha = math.sqrt(1+ustar/2)+math.sqrt(ustar/2)
    sigma = alpha*eta/math.sqrt(2*epsilon)
    return sigma

## LOSS FUNCTIONS

def make_privatizer_loss(map_params, num_grids, batch_size, utility_weights, rho, n):
    def utility_loss(x,y):
        bx = signal_map_params(x,map_params,n)
        by = signal_map_params(y,map_params,n)
        l1 = (bx-by).pow(2).mean()
        l2 = (x-y).pow(2).mean()
        l3 = (y[:,:,[1+3*i for i in range(n)] + [2+3*i for i in range(n)]]-x[:,:,[1+3*i for i in range(n)] + [2+3*i for i in range(n)]]).pow(2).mean()
        # cx,_,_ = density_count(x,num_grids)
        # cy,_,_ = density_count(y, num_grids)
        # l4 = (cx-cy).pow(2).mean()/batch_size
        # l6 = density_loss(x,y, batch_size)
        return l1, l2, l3#, l4, l6
    def privatizer_loss(x,y,u,uhat):
        l1, l2, l3 = utility_loss(x,y)
        # l1, l2, l3, l4, l6 = utility_loss(x,y)
        l = torch.nn.CrossEntropyLoss()
        l5 = l(uhat,u)
        # w1,w2,w3,w4 = utility_weights
        w1,w2,w3 = utility_weights
        # total_loss = rho*(w1*l1+w2*l2+w3*l3+w4*l6)-(1-rho)*l5
        total_loss = rho*(w1*l1+w2*l2+w3*l3)-(1-rho)*l5
        return total_loss
    return utility_loss, privatizer_loss

def make_adversary_loss(privacy_weights, n):
    def adversary_loss(u,x,uhat,lochat):
        l = torch.nn.CrossEntropyLoss()
        dist = (x[:,:,[1+3*i for i in range(n)] + [2+3*i for i in range(n)]]-lochat).pow(2).mean()
        w1, w2 = privacy_weights
        return w1*l(uhat,u)+w2*dist
    return adversary_loss

## ADVERSARY

def make_adversary(num_features, num_units, num_users, num_points_per_entry):
    adversary = torch.nn.Sequential(
        torch.nn.Linear(num_features, num_units),
        torch.nn.ReLU(),
        torch.nn.Linear(num_units, num_units),
        torch.nn.ReLU(),
        # torch.nn.Linear(num_units, num_units),
        # torch.nn.ReLU(),
        torch.nn.Linear(num_units, num_users+2*num_points_per_entry)
    )
    # adversary.apply(init_weights)
    adversary.benchmark = True
    adversary.cuda(CUDA1)
    adversary.double()
    adversary_optimizer = optim.Adam(adversary.parameters(),lr=0.01, betas=(0.9,0.999))
    return adversary, adversary_optimizer

## PRIVATIZER

def make_gap_privatizer(num_features, num_units):
    gap_privatizer = torch.nn.Sequential(
        torch.nn.Linear(num_features, num_units),
        torch.nn.ReLU(),
        torch.nn.Linear(num_units, num_units),
        torch.nn.ReLU(),
        torch.nn.Linear(num_units, num_units),
        torch.nn.ReLU(),
        torch.nn.Linear(num_units, num_features)
    )
    # gap_privatizer.apply(init_weights)
    gap_privatizer.benchmark = True
    gap_privatizer.cuda(CUDA1)
    gap_privatizer.double()
    gap_privatizer_optimizer = optim.Adam(gap_privatizer.parameters(),lr=0.001, betas=(0.9,0.999))
    return gap_privatizer, gap_privatizer_optimizer

def dp_privatizer(x,s, norm_clip):
    normvec = torch.norm(x,p=2,dim=2)
    scalevec = norm_clip/normvec
    scalevec[scalevec>1] = 1
    x = torch.transpose(torch.transpose(x,0,1)*scalevec,0,1).double()
    noise = torch.normal(mean=torch.zeros_like(x),std=s).cuda(CUDA1).double()
    y = x + noise
    return y

def noise_privatizer(x, sigma):
    noise = torch.normal(mean=torch.zeros_like(x),std=sigma).cuda(CUDA1).double()
    y = x + noise
    return y

def make_codebook(codebook_size, batch_size, num_features):
    prob_distr = gaussian_kde(chania_dataset.augmented_data.values.T)
    codebook = {}
    for i in range(codebook_size):
        y = prob_distr.resample(batch_size).T
        y = torch.DoubleTensor(y.reshape(batch_size,1,num_features)).cuda(CUDA1)
        codebook[y] = None
    return codebook

def MI_privatizer(x, codebook, codebook_multiplier, utility_loss):
    for y in codebook.keys():
        loss_utility = utility_loss(x,y)[:-1]
        codebook[y] = sum(loss_utility).item()
    options = list(codebook.keys())
    options.append(x)
    weights = list(map(lambda x: math.e**(-x*codebook_multiplier), codebook.values()))
    weights.append(1)
    best = random.choices(options, weights)[0]
    return best

## TRAIN_SPLIT
def train(num_epochs, train_loader, PRIVATIZER, gap_privatizer, gap_privatizer_optimizer, codebook, codebook_multiplier, utility_loss, privatizer_loss, sigma_dp, norm_clip, sigma_gaussian, adversary_optimizer, adversary, adversary_loss, batch_size, num_users):
    for epoch in range(num_epochs):
        # iterate through the training dataset
        for i, batch in enumerate(train_loader):
            # unpack batch
            x, u = batch['x'], batch['u'].squeeze()
            if x.shape[0] != batch_size:
                break
            # generate privatized batch
            if PRIVATIZER == "gap_privatizer":
                # reset privatizer gradients
                gap_privatizer_optimizer.zero_grad()
                y = gap_privatizer(x)
            elif PRIVATIZER == "MI_privatizer":
                y = MI_privatizer(x, codebook, codebook_multiplier, utility_loss)
            elif PRIVATIZER == "dp_privatizer":
                y = dp_privatizer(x,sigma_dp, norm_clip)
            elif PRIVATIZER == "noise_privatizer":
                y = noise_privatizer(x, sigma_gaussian)
            else:
                raise ValueError

            # reset adversary gradients
            adversary_optimizer.zero_grad()
            estimate = adversary(y).squeeze()
            uhat, lochat = estimate[:,:num_users], estimate[:,num_users:]

            # train adversary
            if PRIVATIZER == "gap_privatizer":
                if i % 1 == 0:
                    aloss = adversary_loss(u,x,uhat,lochat)
                    aloss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(adversary.parameters(), 1000)
                    adversary_optimizer.step()
            else:
                aloss = adversary_loss(u,x,uhat,lochat)
                aloss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), 1000)
                adversary_optimizer.step()

            if PRIVATIZER == "gap_privatizer":
                # evaluate utility loss
                ploss = privatizer_loss(x,y,u,uhat)
                # train privatizer
                ploss.backward()
                torch.nn.utils.clip_grad_norm_(gap_privatizer.parameters(), 1000)
                gap_privatizer_optimizer.step()

            # print progress
            if i % 12 == 0 and i > 0:
                # evaluate utility loss
                aloss = adversary_loss(u,x,uhat,lochat)
                loss_utility = utility_loss(x,y)[:-1]
                print(i+1,"aloss:",aloss.item(),"uloss:",sum(loss_utility).item())
                if PRIVATIZER == "gap_privatizer":
                    print("ploss:",ploss.item())

def test(test_loader, test_epochs, PRIVATIZER, gap_privatizer_optimizer, gap_privatizer, codebook, codebook_multiplier, utility_loss, privatizer_loss, sigma_dp, norm_clip, sigma_gaussian, adversary, map_params, num_grids, batch_size, num_users, n):
    correct = 0
    loc_error = 0
    # l1,l2,l3,l4,l5,l6 = 0,0,0,0,0,0
    l1,l2,l3,l5 = 0,0,0,0

    # iterate through test data
    for epoch in range(test_epochs):
        for i,batch in enumerate(test_loader):

            # unpack batch
            x, u = batch['x'], batch['u'].squeeze()
            if x.shape[0] != batch_size:
                break
            # generate privatized batch
            if PRIVATIZER == "gap_privatizer":
                # reset privatizer gradients
                gap_privatizer_optimizer.zero_grad()
                y = gap_privatizer(x)
            elif PRIVATIZER == "MI_privatizer":
                y = MI_privatizer(x, codebook, codebook_multiplier, utility_loss)
            elif PRIVATIZER == "dp_privatizer":
                y = dp_privatizer(x,sigma_dp, norm_clip)
            elif PRIVATIZER == "noise_privatizer":
                y = noise_privatizer(x, sigma_gaussian)
            else:
                raise ValueError

            # estimate userID and location
            estimate = adversary(y).squeeze()
            uhat, lochat = estimate[:,:num_users], estimate[:,num_users:]

            # Privacy Metric
            _, upred = torch.max(uhat.data,1)
            correct+=(upred==u).sum().item()/batch_size
            loc_error = (x[:,:,[1+3*i for i in range(n)] + [2+3*i for i in range(n)]]-lochat).pow(2).mean().item()

            # Utility Metrics
            # l1, l2, l3, l4, l6 = utility_loss(x,y)
            this_l1, this_l2, this_l3 = utility_loss(x,y)
            l1 += this_l1
            l2 += this_l2
            l3 += this_l3

    # return 100*correct/(i+1)/test_epochs, loc_error/(i+1)/test_epochs, l1.item()/(i+1)/test_epochs, l2.item()/(i+1)/test_epochs, l3.item()/(i+1)/test_epochs, l4.item()/(i+1)/test_epochs
    return 100*correct/(i+1)/test_epochs, loc_error/(i+1)/test_epochs, l1.item()/(i+1)/test_epochs, l2.item()/(i+1)/test_epochs, l3.item()/(i+1)/test_epochs


if __name__ == '__main__':
    FILENAME = r'C:\Users\mclark\Documents\GitHub\pytorch_privacy\clean\daytabase_no_shuffle.csv'

    BATCH_SIZE = 54 # todo
    TRAIN_SPLIT = 0.7

    NUM_POINTS_PER_ENTRY = 150
    NUM_FEATURES = 3*NUM_POINTS_PER_ENTRY
    NUM_UNITS = 512
    NUM_USERS = 9
    NUM_EPOCHS = 50 # todo
    TEST_EPOCHS = 3

    DELTA = 0.00001
    NORM_CLIP=33 #todo

    UTILITY_WEIGHTS = (1,1,1)
    PRIVACY_WEIGHTS =(1,1)
    MAP_PARAMS = 2
    NUM_GRIDS = 16

    CODEBOOK_SIZE = 5

    userID = {
    'a841f74e620f74ec443b7a25d7569545':0,
    '22223276ea84bbce3a62073c164391fd':1,
    '510635002cb29804d54bff664cab52be':2,
    '7cbc37da05801d46e7d80c3b99fd5adb':3,
    '7023889b4439d2c02977ba152d6f4c6e':4,
    '8425a81da55ec16b7f9f80c139c235a2':5,
    '6882f6cf8c72d6324ba7e6bb42c9c7c2':6,
    '1e33db5d2be36268b944359fbdbdad21':7,
    '892d2c3aae6e51f23bf8666c2314b52f':8,
    }

    chania_dataset = ChaniaDataset(csv_file=FILENAME, num_features=NUM_FEATURES, transform=None, normalize=True)
    train_size=int(TRAIN_SPLIT*len(chania_dataset))
    test_size = len(chania_dataset)-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(chania_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # experiment_name = "baseline experiment"
    # print(experiment_name)
    # with open(RESULT_FILENAME, "a") as fd:
    #     fd.write(experiment_name)
    #     fd.write("\n")

    # uncomment one of these chunks to run a test

    # # small multipliers private, large multipliers emphasize utility
    # PRIVATIZER = "MI_privatizer"
    # EPSILON, SIGMA, RHO = 0, 0, 0
    # # for CODEBOOK_MULTIPLIER in [0.001,0.009,0.01,0.09,0.1,0.9,1.0]:
    # for CODEBOOK_MULTIPLIER in [0, 1.0]:

    # PRIVATIZER = "dp_privatizer"
    # SIGMA, RHO, CODEBOOK_MULTIPLIER = 0, 0, 0
    # for EPSILON in [1,2,3,4,5,6,7,8,9,10]:

    PRIVATIZER = "noise_privatizer"
    EPSILON, RHO, CODEBOOK_MULTIPLIER = 0, 0, 0
    # for SIGMA in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for SIGMA in [0]:

    # # rho of 0 is private, 1 is useful
    # PRIVATIZER = "gap_privatizer"
    # EPSILON, SIGMA, CODEBOOK_MULTIPLIER = 0, 0, 0
    # for RHO in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:

        adversary, adversary_optimizer = make_adversary(NUM_FEATURES, NUM_UNITS, NUM_USERS, NUM_POINTS_PER_ENTRY)
        if PRIVATIZER == "gap_privatizer":
            gap_privatizer, gap_privatizer_optimizer = make_gap_privatizer(NUM_FEATURES, NUM_UNITS)
        else:
            gap_privatizer, gap_privatizer_optimizer = None, None
        if PRIVATIZER == "MI_privatizer":
            codebook = make_codebook(CODEBOOK_SIZE, BATCH_SIZE, NUM_FEATURES)
        else:
            codebook = None
        utility_loss, privatizer_loss = make_privatizer_loss(MAP_PARAMS, NUM_GRIDS, BATCH_SIZE, UTILITY_WEIGHTS, RHO, NUM_POINTS_PER_ENTRY)

        adversary_loss = make_adversary_loss(PRIVACY_WEIGHTS, NUM_POINTS_PER_ENTRY)
        sigma_dp = analytical_gaussian_sigma(NORM_CLIP, EPSILON, DELTA)
        train(NUM_EPOCHS, train_loader, PRIVATIZER, gap_privatizer, gap_privatizer_optimizer, codebook, CODEBOOK_MULTIPLIER, utility_loss, privatizer_loss, sigma_dp, NORM_CLIP, SIGMA, adversary_optimizer, adversary, adversary_loss, BATCH_SIZE, NUM_USERS)
        acc, loc_error, map_error, distortion, dist_error = test(test_loader, TEST_EPOCHS, PRIVATIZER, gap_privatizer_optimizer, gap_privatizer, codebook, CODEBOOK_MULTIPLIER, utility_loss, privatizer_loss, sigma_dp, NORM_CLIP, SIGMA, adversary, MAP_PARAMS, NUM_GRIDS, BATCH_SIZE, NUM_USERS, NUM_POINTS_PER_ENTRY)

        RESULT_FILENAME = "trace_"+PRIVATIZER+".csv"
        with open(RESULT_FILENAME, "a") as fd:
            fd.write(PRIVATIZER)
            fd.write(",")
            if PRIVATIZER == "gap_privatizer":
                print(PRIVATIZER,"RHO=",RHO)
                fd.write(str(RHO))
            elif PRIVATIZER == "noise_privatizer":
                print(PRIVATIZER,"SIGMA=",SIGMA)
                fd.write(str(SIGMA))
            elif PRIVATIZER == "dp_privatizer":
                print(PRIVATIZER,"EPSILON=",EPSILON)
                fd.write(str(EPSILON))
            elif PRIVATIZER == "MI_privatizer":
                print(PRIVATIZER,"CODEBOOK SIZE=", CODEBOOK_SIZE)
                fd.write(str(CODEBOOK_SIZE))
            fd.write(",")

            print("Privacy Metrics:", acc, loc_error)
            fd.write(str(acc))
            fd.write(",")
            fd.write(str(loc_error))
            fd.write(",")
            print("Utility Metrics:", map_error, distortion, dist_error)
            fd.write(str(map_error))
            fd.write(",")
            fd.write(str(distortion))
            fd.write(",")
            fd.write(str(dist_error))
            fd.write("\n")
