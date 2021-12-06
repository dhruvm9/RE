#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:12:42 2020

@author: dhruv
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
import os
from wrappers import loadSpikeData
from wrappers import loadXML
from wrappers import loadPosition
from wrappers import loadEpoch
from functions import *
from matplotlib.colors import hsv_to_rgb
import sys
from functions import computeAngularVelocityTuningCurves

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

data_directory = '/media/DataDhruv/Recordings/B0800/B0801/B0801-211119A'
#data_directory = '/media/DataDhruv/Recordings/KA_test'
#data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/SandyReplayAnalysis/Data/A2938-210819'

files = os.listdir(data_directory) 

# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
episodes = ['sleep', 'wake']
# events = ['1', '3']
events = ['1']


spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
position = loadPosition(data_directory, events, episodes)
#position = loadPos_Adrian(data_directory, events, episodes)

wake_ep = loadEpoch(data_directory, 'wake', episodes)
sleep_ep = loadEpoch(data_directory, 'sleep')                    


#sys.exit()

figure()
plot(position['ry'].restrict(wake_ep.loc[[0]]))
title('Raw tracking data')
show()

figure()
plt.plot(position['x'].restrict(wake_ep.loc[[0]]), position['z'].restrict(wake_ep.loc[[0]]))
# plt.axis('off')
#plt.title('Position tracking')
#plt.xlabel('x (m)')
#plt.ylabel('y (m)')
#plt.rc('font', size=BIGGER_SIZE) 
show()

# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



from functions import computeAngularTuningCurves
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60, 120)

from functions import smoothAngularTuningCurves
tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)
SI = computeSpatialInfo(tuning_curves, position['ry'], wake_ep.loc[[0]] )

wake2_ep = splitWake(wake_ep.loc[[0]])
tokeep2 = []
stats2 = []
tcurves2 = []
    
for i in range(2):
        tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
        tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
        
        spatial_curves, extent = computePlaceFields(spikes, position[['x', 'z']], wake2_ep.loc[[i]], 40)
        figure()
        for i in spikes:
            title('Spatial tuning')
            subplot(7,6,i+1)
            plt.rc('font', size=SMALL_SIZE)  
            tmp = scipy.ndimage.gaussian_filter(spatial_curves[i].values, 1)
            imshow(tmp, extent = extent, interpolation = 'bilinear', cmap = 'jet')
            colorbar()
            plt.subplots_adjust(wspace=0.2, hspace=1, top = 0.85)
            
 
        # figure()
        
        # for j, n in enumerate(tuning_curves.columns):
        #     title('Neuron' + ' ' + str(j) + ' shank_' +str(shank[n]) + ' portion' + str(i+1), loc ='center', pad=25)   
        #     subplot(7,6,j+1, projection = 'polar')
        #     plot(tcurves_half[n])
        #     subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        # show()

        
        tokeep, stat = findHDCells(tcurves_half, z = 10, p = 0.05 , m = 1) 
        tokeep2.append(tokeep)
        stats2.append(stat)
        tcurves2.append(tcurves_half)
        
tokeep = np.intersect1d(tokeep2[0], tokeep2[1])


figure()
for i, n in enumerate(tuning_curves.columns):
    #title('Neuron' + ' ' + str(i) + ' shank_' +str(shank[n]) + ' full session', loc ='center', pad=25)   
    subplot(5,5,i+1, projection = 'polar')
    plt.plot(tuning_curves[n])
    plt.xticks([])
    plt.grid(b=None)
    plt.gca().set_yticklabels([])
    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    # plt.axis('off')
    subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
    
show()

velo_curves = computeAngularVelocityTuningCurves(spikes, position['ry'] , wake_ep.loc[[0]], nb_bins = 61, bin_size = 10000, norm=True)
velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

from functions import computeSpeedTuningCurves
speed_curves = computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]], bin_size = 0.1, nb_bins = 20, speed_max = 0.4)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


figure()
for i in spikes:
    title('Angular velocity tuning')
    subplot(7,6,i+1)
    plot(velo_curves[i], label = str(shank[i]))
    legend()

figure()
for i in spikes:
    title('Speed tuning')
    subplot(7,6,i+1)
    plot(speed_curves[i], label = str(shank[i]))
    legend()    

##############################################################################################################################################
##############################################################################################################################################
#Trajectory of mouse, coloured by head direction per spike per neuron
##############################################################################################################################################
##############################################################################################################################################
# position_spike = [0]*len(spikes)
# RGB = [0]*len(spikes)
# position_angle = [0]*len(spikes)

# for n in spikes:
#     #get position of the animal per spike, for each neuron
#     position_tsd = position.restrict(wake_ep.loc[[0]])
#     position_spike[n] = position_tsd.realign(spikes[n].restrict(wake_ep.loc[[0]]))
    
#     #get the direction of the animal's head at the time of the spike
#     position_angle[n] = position_spike[n]['ry'].values/(2*np.pi)
#     HSV = np.vstack((position_angle[n], np.ones_like(position_angle[n]), np.ones_like(position_angle[n]))).T
#     RGB[n] = hsv_to_rgb(HSV)
#     RGB[n][np.isnan(RGB[n])] = 0

# #plot the trajectory of the animal, with the position during a spike coloured by head direction
# n = len(spikes)
# groups = np.array_split(np.arange(n), (n//9)+1)
# for g in range(len(groups)):    
#     figure()
#     for i,n in enumerate(groups[g]):
#         subplot(3,3,i+1)
#         plot(position['x'].restrict(wake_ep.loc[[0]]), position['z'].restrict(wake_ep.loc[[0]]), color = 'grey', alpha= 0.3, linewidth = 0.5)
#         plt.scatter(position_spike[n]['x'], position_spike[n]['z'], alpha = 0.25, s = 0.5, c= RGB[n])   
        
            
#compute autocorrs 

autocorr1 = compute_AutoCorrs(spikes, sleep_ep, binsize = 1, nbins = 600)
autocorr2 = compute_AutoCorrs(spikes, wake_ep, binsize = 1, nbins = 600)

figure()    
for i in spikes:
      # title('Autocorr_neuron' + ' ' + str(i), loc ='center', pad=25)
      subplot(5,5,i+1)
      plt.plot(autocorr1[0][i], label = 'sleep')
      plt.plot(autocorr2[0][i], label = 'wake')
      plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
      legend(loc = 'upper left')
      subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
    
      
plt.figure()
for i in spikes:
      plt.scatter(autocorr2[1][i],autocorr1[1][i], c= 'k')
      plt.title('Mean FR during wake and sleep')
      plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
      plt.xlabel('Wake FR (Hz)')
      plt.ylabel('Sleep FR (Hz)')
      

#mean waveforms 
meanwavef, maxch = loadMeanWaveforms(data_directory)
meanwavef = meanwavef[list(spikes.keys())]
meanwavef.columns = pd.Index(spikes)

allwave = []

for i,n in zip(meanwavef.columns,spikes):
    tmp = meanwavef[i].values.reshape(32,len(meanwavef[i].values)//32) 
    allwave.append(pd.DataFrame(data = tmp[:,maxch[i]], columns = [n]))
  
# figure()    
# for i in spikes:
#      title('Mean Waveforms')
#      subplot(7,6,i+1)
#      plot(allwave[i], label = str(shank[i]))
#      legend() 
#      subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
     
     
     
     
spatial_curves1, extent = computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[0]], 40)
spatial_curves2, extent = computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[1]], 40)

for i in spikes.keys(): 
        
    spatial_curves1[i] = scipy.ndimage.gaussian_filter(spatial_curves1[i].values, 1)
    spatial_curves2[i] = scipy.ndimage.gaussian_filter(spatial_curves2[i].values, 1)

occ = computeOccupancy(position[['x', 'z']],100)
occ = scipy.ndimage.gaussian_filter(occ, 1)

SI_p1  = np.zeros(len(spikes.keys()))
# for i in spikes.keys(): 
#     SI_p1[i] = computeSpatialInfo(np.matrix.flatten(spatial_curves1[i]), occ, wake_ep.loc[[0]] )
    # SI_p2[i] = computeSpatialInfo(spatial_curves2, occ, wake_ep.loc[[1]] )

    
    # pf = np.matrix.flatten(spatial_curves1[i])
    # occupancy, _     = np.histogram(pf, np.matrix.flatten(occ))
    # occ = np.atleast_2d(occupancy/occupancy.sum()).T
    # f = np.sum(pf * occ, 0)
    # pf = pf / f
    # SI = np.sum(occ * pf * np.log2(pf), 0)
    # SI = pd.DataFrame(index = tc.columns, columns = ['SI'], data = SI)

           
n = len(spikes)    
groups = np.array_split(np.arange(n), (n//9)+1)
for g in range(len(groups)):    
    figure()
    for i,n in enumerate(groups[g]):
        title('Spatial tuning first half')    
        subplot(3,3,i+1)
        tmp = scipy.ndimage.gaussian_filter(spatial_curves1[n].values, 1)
        #tmp2 = scipy.ndimage.gaussian_filter(spatial_curves2[i].values, 1.2)
        imshow(tmp, extent = extent, interpolation = 'bilinear', cmap = 'jet')
        #imshow(tmp2, extent = extent, interpolation = 'bilinear', cmap = 'jet')
        colorbar()
        plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        
    figure()
    for i,n in enumerate(groups[g]):
        title('Spatial tuning second half')    
        subplot(3,3,i+1)
        tmp2 = scipy.ndimage.gaussian_filter(spatial_curves2[n].values, 1)
        imshow(tmp2, extent = extent, interpolation = 'bilinear', cmap = 'jet')
        colorbar()
        plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        
        
        
# tcurves1 = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 121)
# tcurves1 = smoothAngularTuningCurves(tcurves1, 10, 2)
# tcurves2 = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[1]], 121)
# tcurves2 = smoothAngularTuningCurves(tcurves2, 10, 2)

# n = len(spikes)    
# groups = np.array_split(np.arange(n), (n//9)+1)
# for g in range(len(groups)):    
#     figure()
#     for i,n in enumerate(groups[g]):
#         title('HD tuning: radial maze')    
#         subplot(3,3,i+1, projection = 'polar')
#         plot(tcurves1[n])
#         plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        
#     figure()
#     for i,n in enumerate(groups[g]):
#         title('HD tuning: open field')    
#         subplot(3,3,i+1, projection = 'polar') 
#         plot(tcurves2[n])
#         plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        
# t,pvalue = mannwhitneyu(SI_arena, SI_KA)
# means_RE = np.nanmean(SI_arena)
# means_AD = np.nanmean(SI_KA)

# label = ['ADn v/s RE']
# x = np.arange(len(label))  # the label locations
# width = 0.35  # the width of the bars


# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, means_RE, width, label='RE')
# rects2 = ax.bar(x + width/2, means_AD, width, label='ADn')

# #pval = np.hstack([(SI_7503), (SI_KA)])

# x2 = [x-width/2, x+width/2]
# #plt.plot(x2, np.vstack(pval.T), 'o-', color = 'k')
# plt.plot(x2[0], SI_arena.T,'o-',color = 'k')
# plt.plot(x2[1], SI_KA.T,'o-',color = 'k')


# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Bits per spike')
# ax.set_title('HD Info for ADn and RE cells')
# ax.set_xticks(x)
# ax.set_xticklabels(label)
# ax.legend(loc = 'upper right')

# fig.tight_layout()


