#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:39:35 2021

@author: dhruv
"""
import numpy as np
import pandas as pd
import neuroseries as nts
import scipy.io
import scipy.stats
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
import seaborn as sns 
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu 
from functions import computeSpeedTuningCurves
from matplotlib.cm import ScalarMappable
    

data_directory = '/media/DataDhruv/Recordings/B0800/B0801'
datasets = np.loadtxt(os.path.join(data_directory,'VTE_dataset.list'), delimiter = '\n', dtype = str, comments = '#')

BIGGER_SIZE = 14

allz1 = []
alltrialnumbers = []
allcorrs = []
allp = []
allwilxp = []
allpcomp = []
alldurcorr = []
alldurp = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    episodes = ['sleep', 'wake']
    events = ['1']

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(path)
    position = loadPosition(path, events, episodes)
    wake_ep = loadEpoch(path, 'wake', episodes)
    sleep_ep = loadEpoch(path, 'sleep')        
    
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)  
    passes = pd.read_csv(path + '/' + s +'_vte.csv')
    
    # if s == 'A8504-210707':
    #     sys.exit()
    
    pass_ep = nts.IntervalSet(start = passes['start'], end = passes['end'])
    
    file = [f for f in listdir if 'VTE' in f]
    vtedata = scipy.io.loadmat(os.path.join(filepath,file[0]))

    xpos = vtedata['x']
    ypos = vtedata['y']

    x1_py = vtedata['x1_py']
    y1_py = vtedata['y1_py']
    v_py = vtedata['v_py']
    r_py = vtedata['r_py']

    z1 = vtedata['z1']
    
    trialnumber = []
    
    for i in range(len(z1[0])):
        allz1.append(z1[0][i])
        trialnumber.append(i)
        alltrialnumbers.append(i)

    v = {}
    x = {}
    y = {}
    r = {}

    for i in range(size(v_py)): 
        v_array = []
        x_array = []
        y_array = []
        r_array = []
    
        for j in range(size(v_py[0][i])):
            v_array.append(v_py[0][i][j][0])
            x_array.append(x1_py[0][i][j][0])
            y_array.append(y1_py[0][i][j][0])
            r_array.append(r_py[0][i][j][0])
    
        v[i] = v_array
        x[i] = x_array
        y[i] = y_array
        r[i] = r_array

    vmax = [] 
    vmin = []

    for i in range(len(v)): 
        vmax.append(max(v[i]))
        vmin.append(min(v[i]))

    miv = min(vmin)
    mav = max(vmax)    
    z2 = z1[0].copy()
    
    #COMPUTE MEAN FIRING RATE
               
    rates = computeMeanFiringRate(spikes, [pass_ep, wake_ep], ['pass','wake'])
    corr, pvalue = pearsonr(rates['wake'],rates['pass'])
    
    dur = np.zeros(len(pass_ep))
    for i in range(len(pass_ep)):
        dur[i] = pass_ep.loc[[i]].tot_length('s')
        
    passrate = pd.DataFrame(index = range(len(pass_ep)) , columns = spikes.keys() )
          
    
    for i in range(len(pass_ep)):
        for j in spikes.keys() :
    #     # plt.figure()
    #     # plt.axis('off')
    #     # plt.plot(xpos,ypos,'silver', zorder = 1)
    #     # plt.plot(position['x'].restrict(pass_ep.loc[[i]]), position['z'].restrict(pass_ep.loc[[i]]))
            r2 = len(spikes[j].restrict(pass_ep.loc[[i]]))/pass_ep.loc[[i]].tot_length('s')
            passrate[j][i] = r2

    
    # sys.exit()
    
    ###COMPUTE SPEED
    bin_size = 1000000 #us
    
    time_bins = np.arange(position.index[0], position.index[-1]+(bin_size), bin_size)
    index = np.digitize(position.index.values, time_bins)
    tmp = position.groupby(index).mean()
    tmp.index = time_bins[np.unique(index)-1]+(bin_size/2)
    distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
    speed = nts.Tsd(t = tmp.index.values[0:-1], d = (distance*100))
    
    
    
    
    
    
 ####FIRING RATE DURING THE COURSE OF A PASS 
         

    # for i in range(len(pass_ep)):    
    #     ep = pass_ep.loc[[i]]
    #     ep = nts.IntervalSet(start = passes.iloc[i]['start']-5e6, end = passes.iloc[i]['end']+5e6)
    #     bins = np.arange(ep.iloc[0,0]/1e6, ep.iloc[0,1]/1e6, bin_size/1e6)       
    #     r = np.zeros((len(bins)-1))
        
    #     sp = speed.restrict(ep)
    
    #     for j in spikes.keys():   
    #         rate_course = []
    #         tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
    #         r = r + tmp
    #         rate_course.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
    #         rate_course = pd.concat(rate_course)
    #         r2 = rate_course.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)    
            
    #         r2 = r2/max(r2)
    #         s1 = sp.values/max(sp.values)
    #         s2 = sp.index/bin_size
                      
            
    #         x0 = 5
    #         x1 = len(r2)-6
        
    #         plt.figure()
    #         plt.title('Neuron ' + str(j+1) + ' Pass ' + str(i+1))
            
    #         plt.plot(r2.index[:x0+1],r2.iloc[:x0+1],'k')
    #         plt.plot(r2.index[x0:x1+1],r2.iloc[x0:x1+1],label = 'Normalized FR')
    #         plt.plot(r2.index[x1:],r2.iloc[x1:],'k')
            
    #         plt.plot(s2[:x0+1],s1[:x0+1],'k')
    #         plt.plot(s2[x0:x1+1],s1[x0:x1+1],label = 'Normalized speed')
    #         plt.plot(s2[x1:],s1[x1:],'k')
    #         plt.legend(loc = 'upper right')
            
    #         plt.ioff()
    #         plt.savefig(path + '/Figs/' + '/Neuron ' + str(j+1) + ' Pass ' + str(i+1) + '.png' )
    
    ###Coded as zIdPhi
    
    # j = 2
    # plt.figure()
    # plt.title('Neuron ' + str(j+1))  
                
    # for i in range(len(pass_ep)):    
    #     ep = pass_ep.loc[[i]]
    #     ep = nts.IntervalSet(start = passes.iloc[i]['start']-5e6, end = passes.iloc[i]['end']+5e6)
    #     bins = np.arange(ep.iloc[0,0]/1e6, ep.iloc[0,1]/1e6, bin_size/1e6)   
        
       
        
    #     r = np.zeros((len(bins)-1))
               
    #     rate_course = []
    #     tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
    #     r = r + tmp
    #     rate_course.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
    #     rate_course = pd.concat(rate_course)
    #     r2 = rate_course.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)    
                   
    #         #Normalizing
    #     bs = (r2.index - min(r2.index)) / (max(r2.index) - min(r2.index))             
    #     normalize = matplotlib.colors.Normalize(vmin=min(z1[0]), vmax=max(z1[0]))
    #     scalarMap = ScalarMappable(norm=normalize, cmap="hot")        
    
    #     x0 = 5
    #     x1 = len(r2)-6
    
    #     plt.scatter(bs,r2, c = np.repeat(z1[0][i],len(r2)),norm = normalize, cmap = 'hot')
    #     plt.plot(bs,r2,c=scalarMap.to_rgba(z1[0][i]))
                    
    #     # plt.scatter(bs[:x0+1],r2.iloc[:x0+1],color = 'k')
    #     # plt.plot(bs[:x0+1],r2.iloc[:x0+1],color = 'k')
    #     # plt.scatter(bs[x0:x1+1],r2.iloc[x0:x1+1], c = np.repeat(z1[0][i],len(r2.iloc[x0:x1+1])),norm = normalize, cmap = 'hot')
    
    #     if i == len(pass_ep)-1:
    #         plt.colorbar(label = 'zIdPhi')
    #         plt.clim(min(z1[0]),max(z1[0]))
                
    #     # plt.plot(bs[x0:x1+1],r2.iloc[x0:x1+1],c=scalarMap.to_rgba(z1[0][i]))
    #     plt.xlabel('Normalized time')
    #     plt.ylabel('FR (Hz)')
    
    
    #     # plt.scatter(bs[x1:],r2.iloc[x1:],color = 'k')
    #     # plt.plot(bs[x1:],r2.iloc[x1:],color = 'k')
        
        
        
 #PETH Centered around pass start and end 
    binsize = 200
    nbins = 100
    neurons = list(spikes.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    
    ix_vte = np.where(z1[0] >= 0.5)
    ix_other = np.where(z1[0] < 0.5)
    
    # tsd_pass = pass_ep.as_units('ms').start.values
    tsd_pass = pass_ep.as_units('ms').end.values
   
#UP State    
    ep_pass = nts.IntervalSet(start = pass_ep.start[0], end = pass_ep.end.values[-1])
               
    rates = []
    
    for i in neurons:
        spk2 = spikes[i].restrict(ep_pass).as_units('ms').index.values
        tmp = crossCorr(tsd_pass[ix_vte], spk2, binsize, nbins)
        # tmp = crossCorr(tsd_pass[ix_other], spk2, binsize, nbins)
        fr = len(spk2)/ep_pass.tot_length('s')
        rates.append(fr)
        cc[i] = tmp
        cc[i] = tmp/fr

    order = np.argsort(cc.loc[0])
    dd = cc[order]
    
        
    fig, ax = plt.subplots()
        #cax = ax.imshow(finalRates.T,extent=[-250 , 150, len(interneuron) , 1],aspect = 'auto', cmap = 'hot')
    cax = ax.imshow(dd.T,extent=[-10000 , 10000, len(neurons) , 1],aspect = 'auto', cmap = 'hot')
        # plt.imshow(finalRates.T,extent=[-250 , 250, len(neurons) , 1],aspect = 'auto', cmap = 'hot')        
        # plt.imshow(finalRates.T,extent=[-250 , 250, len(pyr) , 1],aspect = 'auto', cmap = 'hot')        
    cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3], label = 'Normalized FR')
    cbar.ax.set_yticklabels(['0', '1', '2', '3'])
    cax.set_clim([0, 3])
    # plt.title('Event-related Xcorr, aligned to VTE pass start_' + s)
    plt.title('Event-related Xcorr, aligned to VTE pass end_' + s)
    ax.set_ylabel('Neuron number')
    ax.set_xlabel('Lag (ms)')

   

 
 
 
 