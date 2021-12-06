# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:56:40 2020

@author: Dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import ipyparallel
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 

#load csv files
data_directory = '/media/DataDhruv/Recordings/A8500/A8504/A8504-210706a'
files = os.listdir(data_directory) 
episodes = ['sleep', 'wake']
events = ['1']

spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
position = loadPosition(data_directory, events, episodes)

wake_ep = loadEpoch(data_directory, 'wake', episodes)
sleep_ep = loadEpoch(data_directory, 'sleep')                    

passes = pd.read_csv(data_directory + '/A8504-210706a_vte.csv')
pass_ep = nts.IntervalSet(start = passes['start'], end = passes['end'])

allidphi = []

for i in range(len(pass_ep)):

    x = np.array(position['x'].restrict(pass_ep.loc[[i]]))
    y = np.array(position['z'].restrict(pass_ep.loc[[i]]))
    x_err = np.random.uniform(-0.01,0.01,(len(x)),)

    dx = np.diff(x)*120
    dy = np.diff(y)*120
    r = np.sqrt(np.square(dx) + np.square(dy))

    T = 1/120
    N = len(x)

    v_x = [] 
    v_y = []
    nvals_x = []
    nvals_y = []
    dphi = []
    nvals_phi = [] 

    t = np.arctan2(dy,dx)
    
    for k in np.arange(1, N): 
        for n in np.arange(1, N-k): 
            y_k = x[-k]
            y_k_n = x[-k-n]
                            
            a = ((k*y_k_n) + ((n-k)*y_k))/n
            b = (y_k - y_k_n)/(n*T)
            linevalues = a + b*T*(k - np.arange(1,n+1))
        
            diff = np.abs(np.flip(x[-k-n:-k]) - linevalues)    
            errorvals = np.flip(x_err[-n:])
            comp = np.less_equal(diff, errorvals)
        
            if any(np.logical_not(comp)) == True:
                nvals_x.append(n) 
                print(n,k) 
                break
   
        v_x.append(b)
    
    for k in np.arange(1, N): 
        for n in np.arange(1, N-k): 
            y_k = y[-k]
            y_k_n = y[-k-n]
                            
            a = ((k*y_k_n) + ((n-k)*y_k))/n
            b = (y_k - y_k_n)/(n*T)
            linevalues = a + b*T*(k - np.arange(1,n+1))
        
            diff = np.abs(np.flip(y[-k-n:-k]) - linevalues)    
            errorvals = np.flip(x_err[-n:])
            comp = np.less_equal(diff, errorvals)
        
            if any(np.logical_not(comp)) == True:
                nvals_y.append(n) 
                print(n,k) 
                break
   
        v_y.append(b)

    v_x = np.flip(v_x[-N:])
    v_y = np.flip(v_y[-N:])

    theta = np.arctan2(v_y,v_x)
    tmp = np.unwrap(theta)

    for k in np.arange(1, N): 
        for n in np.arange(1, N-k): 
            y_k = tmp[-k]
            y_k_n = tmp[-k-n]
            #y_k = theta[-k]
            #y_k_n = theta[-k-n]
                        
            a = ((k*y_k_n) + ((n-k)*y_k))/n
            b = (y_k - y_k_n)/(n*T)
            linevalues = a + b*T*(k - np.arange(1,n+1))
        
            diff = np.abs(np.flip(theta[-k-n:-k]) - linevalues)    
            errorvals = np.flip(x_err[-n:])
            comp = np.less_equal(diff, errorvals)
        
            if any(np.logical_not(comp)) == True:
                nvals_phi.append(n) 
                print(n,k) 
                break
   
        dphi.append(b)

    dphi = np.flip(dphi[-N:])
    
    dphi = np.abs(dphi)

    idphi = np.sum(dphi)
    allidphi.append(idphi)    



# plt.title('Position tracking')
# for i in range(len(pass_ep)):
#     plt.figure()
#     plt.plot(position['x'].restrict(pass_ep.loc[[i]]), position['z'].restrict(pass_ep.loc[[i]]),'o')
#     plt.xlim(-0.3,0.2)
#     plt.ylim(-0.2,0.3)



