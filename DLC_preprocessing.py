#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:46:49 2021

@author: dhruv
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

# #get nose, leftear and rightear data
tracking_data =  pd.read_hdf('A2929-200714.h5', mode = 'r')
hd_cols = [0,1,3,4,6,7]
hd_data = tracking_data.iloc[:,hd_cols]

# #compute centroid
x_cols = [0,2,4]
y_cols = [1,3,5]
all_x_coords = hd_data.iloc[:,x_cols]
all_y_coords = hd_data.iloc[:,y_cols]
length = all_x_coords.iloc[0,:].shape[0]

x_sum = all_x_coords.sum(axis = 1)
y_sum = all_y_coords.sum(axis = 1)

x_cent = x_sum/length
y_cent = y_sum/length

hd_centroid = np.zeros((len(x_cent),2))
hd_centroid[:,0] = x_cent 
hd_centroid[:,1] = y_cent

#plotting  of HD centroids
t = range(len(hd_data))
plt.figure()
plt.scatter(x_cent, y_cent, c = t)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title('HD tracking by DeepLabCut')
plt.ylabel('y-coordinate')
plt.xlabel('x-coordinate')

#get data for body length 
tailbase_cols = [15,16]
tailbase_coords = np.zeros((len(tracking_data),2))
tailbase_coords[:,0] = tracking_data.iloc[:,tailbase_cols[0]]
tailbase_coords[:,1] = tracking_data.iloc[:,tailbase_cols[1]]

#compute distance between centroid and tail base to get body length 
dist = np.sqrt(np.square((tailbase_coords[:,0] - hd_centroid[:,0])) + np.square((tailbase_coords[:,1] - hd_centroid[:,1])))

#compute dx, dy, r and theta values 
x = hd_centroid[:,0]
y = hd_centroid[:,1]
x_err = np.random.uniform(0.1,4,(len(x)),)

T = 1/120
N = len(x)

v_x = [] 
v_y = []
nvals_x = []
nvals_y = []
dphi = []
nvals_phi = [] 


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


dx = np.diff(x)*120
dy = np.diff(y)*120
#r = np.sqrt(np.square(dx) + np.square(dy))

v_x = np.flip(v_x[-N:])
v_y = np.flip(v_y[-N:])

theta = np.arctan2(v_y,v_x)
t = np.arctan2(dy,dx)

for k in np.arange(1, N): 
    for n in np.arange(1, N-k): 
        y_k = theta[-k]
        y_k_n = theta[-k-n]
                        
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

#plot r and body length 
#plt.figure()
#plt.subplot(211)
#plt.plot(r)
#plt.title('Displacement per frame')
#plt.ylabel('r value')
#plt.xlabel('frame number')
#plt.subplot(212)
#plt.plot(dist)
#plt.title('Body length')
#plt.ylabel('Distance between tail base and HD centroid')
#plt.xlabel('frame number')

#plt.figure()
#plt.plot(dx[0:N-1])
#plt.plot(v_x)

#plt.figure()
#plt.plot(dy[0:N-1])
#plt.plot(v_y)

#plt.figure()
#plt.plot(t)
#plt.plot(theta)