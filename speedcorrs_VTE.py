#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:04:47 2021

@author: dhruv
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:12:07 2021

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
from scipy.stats import pearsonr, wilcoxon 
from functions import computeSpeedTuningCurves
    

data_directory = '/media/DataDhruv/Recordings/A8500/A8504'
datasets = np.loadtxt(os.path.join(data_directory,'VTE_dataset.list'), delimiter = '\n', dtype = str, comments = '#')

BIGGER_SIZE = 14

allz1 = []
alltrialnumbers = []
allcorrs = []
allpvals = []
allvcorrs = []
allvp = []
allwilxp = []

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
    normalize = matplotlib.colors.Normalize(vmin=miv, vmax=mav)
    
    #COMPUTE MEAN FIRING RATE AND AVERAGE SPEED DURING A PASS
    
    rates = computeMeanFiringRate(spikes, [pass_ep, wake_ep], ['pass','wake'])
    corr, pvalue = pearsonr(rates['wake'],rates['pass'])
    
    avg_v = np.zeros(len(v))
    for i in range(len(v)):
        avg_v[i] = np.mean(v[i])
   
    #COMPUTE FIRING RATE  DURING A PASS FOR EACH CELL
    passrate = pd.DataFrame(index = range(len(pass_ep)) , columns = spikes.keys() )
      
    for i in range(len(pass_ep)):
        for j in spikes.keys() :
            # plt.figure()
            # plt.axis('off')
            # plt.plot(xpos,ypos,'silver', zorder = 1)
            # plt.plot(position['x'].restrict(pass_ep.loc[[i]]), position['z'].restrict(pass_ep.loc[[i]]))
            r2 = len(spikes[j].restrict(pass_ep.loc[[i]]))/pass_ep.loc[[i]].tot_length('s')
            passrate[j][i] = r2
            
    #CORRELATION BETWEEN PASS FR AND zIdPhi
    sessioncorrs = []
    sessionvcorrs = []
    sessionp = []
    sessionvp = []
    
    for i in spikes.keys():
        corr, pvalue = pearsonr(passrate[i],z1[0])
        plt.figure()
        # plt.title('Corr between pass FR v/s zIdPhi_Neuron_' + str(i+1) +'_' + s)
        # plt.scatter(z1[0],passrate[i], label = 'R =  ' +  str(round(corr,4)))
        # plt.legend(loc = 'upper right')
        # plt.ylabel('Pass FR (Hz)')
        # plt.xlabel('zIdPhi')
        # plt.xticks([-1, 0, 1, 2])
        # sessioncorrs.append(corr)
        sessionp.append(pvalue)
        allcorrs.append(corr)
        allpvals.append(pvalue)
        
        vcorr, vp = pearsonr(passrate[i],avg_v)
        sessionvcorrs.append(vcorr)
        sessionvp.append(vp)
        allvcorrs.append(vcorr)
        allvp.append(vp)
        
        # plt.figure()
        # plt.title('Corr between pass speed v/s pass FR_Neuron_' + str(i+1) +'_' + s)
        # plt.scatter(passrate[i],avg_v, label = 'R =  ' +  str(round(vcorr,4)))
        # plt.legend(loc = 'upper right')
        # plt.ylabel('Pass FR (Hz)')
        # plt.xlabel('Velocity (cm/s)')
        
        
        
        
        
    
    
    # plt.figure()
    # plt.hist(sessioncorrs, label = 'Mean =' + str(round(np.mean(sessioncorrs),4)))
    # plt.title('Distribution of zIdPhi v/s FR corr_' + s )
    # plt.xlabel('Pearson R')
    # plt.ylabel('Number of cells')
    # plt.legend(loc = 'upper right')
    
    z_statistic, p_value = wilcoxon(np.array(sessioncorrs).flatten()-0)
    allwilxp.append(p_value)
    
   # ###PASS FIRING RATE
   #  bin_size = (1/120)*1e6 #s
    
   #  course_rate = pd.DataFrame(index = range(len(pass_ep)) , columns = spikes.keys() ) 
   #  timestamps = {}

   #  for i in range(len(pass_ep)):
   #      ep = nts.IntervalSet(start = passes.iloc[i]['start'], end = passes.iloc[i]['end'])
   #      bins = np.arange(ep.iloc[0,0]/1e6, ep.iloc[0,1]/1e6, bin_size/1e6)       
        
   #      for j in spikes.keys():
   #         rate_pass = []
   #         tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
   #         rate_pass.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
   #         rate_pass = pd.concat(rate_pass)
   #         r3 = rate_pass.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
   #         course_rate[j][i] = r3.values
   #         timestamps[i] = r3.index
        
    
        
   
        
        
    
    
   
    
    ####FIRING RATE DURING BEFORE, AFTER AND DURING PASS
    # bin_size = 1000000/120 #us
    # rate_course = []

    # #ep = pass_ep.loc[[0]]
    # ep = nts.IntervalSet(start = passes.iloc[0]['start']-5e6, end = passes.iloc[0]['end']+5e6)
    # bins = np.arange(ep.iloc[0,0]/1e6, ep.iloc[0,1]/1e6, bin_size/1e6)       
    # #r = np.zeros((len(bins)-1))
    
    # j = 2   
    # tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
    # #r = r + tmp
    # rate_course.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
    # rate_course = pd.concat(rate_course)
    # r2 = rate_course.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)    
    # r2 = r2/max(r2)
    # x0 = 5
    # x1 = len(r2)-6
    
    # plt.figure()
    # plt.plot(r2.index[:x0+1],r2.iloc[:x0+1],'k')
    # plt.plot(r2.index[x0:x1+1],r2.iloc[x0:x1+1])
    # plt.plot(r2.index[x1:],r2.iloc[x1:],'k')
    
    
    # j = 14  
    # rate_course = []
    # tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
    # #r = r + tmp
    # rate_course.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
    # rate_course = pd.concat(rate_course)
    # r2 = rate_course.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)    
    # r2 = r2/max(r2)
    # x0 = 5
    # x1 = len(r2)-6
    
    # plt.figure()
    # plt.plot(r2.index[:x0+1],r2.iloc[:x0+1],'k')
    # plt.plot(r2.index[x0:x1+1],r2.iloc[x0:x1+1])
    # plt.plot(r2.index[x1:],r2.iloc[x1:],'k')
    
    # from functions import computeSpeedTuningCurves
    # speed_curves = computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]], bin_size = 0.1, nb_bins = 20, speed_max = 0.4)
    # speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
    
    # figure()
    # for i in spikes:
    #     title('Speed tuning')
    #     subplot(7,6,i+1)
    #     plot(speed_curves[i], label = str(shank[i]))
    #     legend()    
    
    # #SAME BUT FOR NON-VTE EXAMPLE
    # bin_size = 1000000 #us
    # rate_course = []

    # #ep = pass_ep.loc[[0]]
    # ep = nts.IntervalSet(start = passes.iloc[22]['start']-5e6, end = passes.iloc[22]['end']+5e6)
    # bins = np.arange(ep.iloc[0,0]/1e6, ep.iloc[0,1]/1e6, bin_size/1e6)       
    # #r = np.zeros((len(bins)-1))
    
    # j = 2   
    # tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
    # #r = r + tmp
    # rate_course.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
    # rate_course = pd.concat(rate_course)
    # r2 = rate_course.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)    
    # r2 = r2/max(r2)
    # x0 = 5
    # x1 = len(r2)-6
    
    # plt.figure()
    # plt.plot(r2.index[:x0+1],r2.iloc[:x0+1],'k')
    # plt.plot(r2.index[x0:x1+1],r2.iloc[x0:x1+1])
    # plt.plot(r2.index[x1:],r2.iloc[x1:],'k')
    
    
    # j = 14  
    # rate_course = []
    # tmp = np.histogram(spikes[j].restrict(ep).index.values/1e6, bins)[0]
    # #r = r + tmp
    # rate_course.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = tmp))
    # rate_course = pd.concat(rate_course)
    # r2 = rate_course.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)    
    # r2 = r2/max(r2)
    # x0 = 5
    # x1 = len(r2)-6
    
    # plt.figure()
    # plt.plot(r2.index[:x0+1],r2.iloc[:x0+1],'k')
    # plt.plot(r2.index[x0:x1+1],r2.iloc[x0:x1+1])
    # plt.plot(r2.index[x1:],r2.iloc[x1:],'k')
    
    # from functions import computeSpeedTuningCurves
    # speed_curves = computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]], bin_size = 0.1, nb_bins = 20, speed_max = 0.4)
    # speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
    
    # figure()
    # for i in spikes:
    #     title('Speed tuning')
    #     subplot(7,6,i+1)
    #     plot(speed_curves[i], label = str(shank[i]))
    #     legend()    
    
    # sys.exit()