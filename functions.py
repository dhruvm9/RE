import numpy as np
from numba import jit
import pandas as pd
import neuroseries as nts
import sys, os
import scipy.stats
from wrappers import *
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages

'''
Utilities functions
Feel free to add your own
'''


#########################################################
# CORRELATION
#########################################################
@jit(nopython=True)
def crossCorr(t1, t2, binsize, nbins):
    ''' 
        Fast crossCorr 
    '''
    nt1 = len(t1)
    nt2 = len(t2)
    if np.floor(nbins/2)*2 == nbins:
        nbins = nbins+1

    m = -binsize*((nbins+1)/2)
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m+j*binsize

    w = ((nbins/2) * binsize)
    C = np.zeros(nbins)
    i2 = 1

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2+1
        while i2 > 1 and t2[i2-1] > lbound:
            i2 = i2-1

        rbound = lbound
        l = i2
        for j in range(nbins):
            k = 0
            rbound = rbound+binsize
            while l < nt2 and t2[l] < rbound:
                l = l+1
                k = k+1

            C[j] += k

    # for j in range(nbins):
    # C[j] = C[j] / (nt1 * binsize)
    C = C/(nt1 * binsize/1000)

    return C

def crossCorr2(t1, t2, binsize, nbins):
    '''
        Slow crossCorr
    '''
    window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
    allcount = np.zeros(nbins+1)
    for e in t1:
        mwind = window + e
        # need to add a zero bin and an infinite bin in mwind
        mwind = np.array([-1.0] + list(mwind) + [np.max([t1.max(),t2.max()])+binsize])    
        index = np.digitize(t2, mwind)
        # index larger than 2 and lower than mwind.shape[0]-1
        # count each occurences 
        count = np.array([np.sum(index == i) for i in range(2,mwind.shape[0]-1)])
        allcount += np.array(count)
    allcount = allcount/(float(len(t1))*binsize / 1000)
    return allcount

def xcrossCorr_slow(t1, t2, binsize, nbins, nbiter, jitter, confInt):        
    times             = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    H0                 = crossCorr(t1, t2, binsize, nbins)    
    H1                 = np.zeros((nbiter,nbins+1))
    t2j                 = t2 + 2*jitter*(np.random.rand(nbiter, len(t2)) - 0.5)
    t2j             = np.sort(t2j, 1)
    for i in range(nbiter):            
        H1[i]         = crossCorr(t1, t2j[i], binsize, nbins)
    Hm                 = H1.mean(0)
    tmp             = np.sort(H1, 0)
    HeI             = tmp[int((1-confInt)/2*nbiter),:]
    HeS             = tmp[int((confInt + (1-confInt)/2)*nbiter)]
    Hstd             = np.std(tmp, 0)

    return (H0, Hm, HeI, HeS, Hstd, times)

def xcrossCorr_fast(t1, t2, binsize, nbins, nbiter, jitter, confInt):        
    times             = np.arange(0, binsize*(nbins*2+1), binsize) - (nbins*2*binsize)/2
    # need to do a cross-corr of double size to convolve after and avoid boundary effect
    H0                 = crossCorr(t1, t2, binsize, nbins*2)    
    window_size     = 2*jitter//binsize
    window             = np.ones(window_size)*(1/window_size)
    Hm                 = np.convolve(H0, window, 'same')
    Hstd            = np.sqrt(np.var(Hm))    
    HeI             = np.NaN
    HeS             = np.NaN    
    return (H0, Hm, HeI, HeS, Hstd, times)    

def compute_AutoCorrs(spks, ep, binsize = 5, nbins = 200):
    # First let's prepare a pandas dataframe to receive the data
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2    
    autocorrs = pd.DataFrame(index = times, columns = np.arange(len(spks)))
    firing_rates = pd.Series(index = np.arange(len(spks)))

    # Now we can iterate over the dictionnary of spikes
    for i in spks:
        # First we extract the time of spikes in ms during wake
        spk_time = spks[i].restrict(ep).as_units('ms').index.values
        # Calling the crossCorr function
        autocorrs[i] = crossCorr(spk_time, spk_time, binsize, nbins)
        # Computing the mean firing rate
        firing_rates[i] = len(spk_time)/ep.tot_length('s')

    # We can divide the autocorrs by the firing_rates
    autocorrs = autocorrs / firing_rates

    # And don't forget to replace the 0 ms for 0
    autocorrs.loc[0] = 0.0
    return autocorrs, firing_rates

#########################################################
# VARIOUS
#########################################################
def computeAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = pd.DataFrame(index = idx, columns = np.arange(len(spikes)))    
    angle             = angle.restrict(ep)
 
    # Smoothing the angle here
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    angle            = nts.Tsd(tmp2%(2*np.pi))
    for k in spikes:
        spks             = spikes[k]
        # true_ep         = nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))        
        spks             = spks.restrict(ep)    
        angle_spike     = angle.restrict(ep).realign(spks)
        spike_count, bin_edges = np.histogram(angle_spike, bins)
        occupancy, _     = np.histogram(angle, bins)
        spike_count     = spike_count/(occupancy+1)
        tuning_curves[k] = spike_count*frequency    

    return tuning_curves


def findHDCells(tuning_curves, z = 50, p = 0.0001 , m = 1):
    """
        Peak firing rate larger than 1
        and Rayleigh test p<0.001 & z > 100
    """
    cond1 = tuning_curves.max()>m
    from pycircstat.tests import rayleigh
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
    for k in tuning_curves:
        stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
    cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
    tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
    
    return tokeep, stat

def decodeHD(tuning_curves, spikes, ep, px, bin_size = 200):
    """
        See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
        tuning_curves: pd.DataFrame with angular position as index and columns as neuron
        spikes : dictionnary of spike times
        ep : nts.IntervalSet, the epochs for decoding
        bin_size : in ms (default:200ms)
        px : Occupancy. If None, px is uniform
    """        
    if len(ep) == 1:
        bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
    else:
        print("TODO, more than one epoch")
        sys.exit()
    
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = tuning_curves.columns)
    for k in spike_counts.columns:
        spks = spikes[k].restrict(ep).as_units('ms').index.values
        spike_counts[k], _ = np.histogram(spks, bins)

    tcurves_array = tuning_curves.values
    spike_counts_array = spike_counts.values
    proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

    part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))    
    part2 = px
    
    for i in range(len(proba_angle)):
        part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
        p = part1 * part2 * part3
        proba_angle[i] = p/p.sum() # Normalization process here

    proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)    
    # proba_angle = proba_angle.astype('float')        
    decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
    return decoded, proba_angle

def computePlaceFields(spikes, position, ep, nb_bins = 20, frequency = 120.0):
    place_fields = {}
    position_tsd = position.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)    
    for n in spikes:
        position_spike = position_tsd.realign(spikes[n].restrict(ep))
        spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
        occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
        mean_spike_count = spike_count/(occupancy+1)
        place_field = mean_spike_count*frequency    
        place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
        
    extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
    return place_fields, extent

def computeOccupancy(position_tsd, nb_bins = 100):
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]    
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    return occupancy

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        padded     = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi),
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi))),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)        
        tuning_curves[i] = smoothed[tcurves.index]

    return tuning_curves

def computeMeanFiringRate(spikes, epochs, name):
    mean_frate = pd.DataFrame(index = spikes.keys(), columns = name)
    for n, ep in zip(name, epochs):
        for k in spikes:
            mean_frate.loc[k,n] = len(spikes[k].restrict(ep))/ep.tot_length('s')
    return mean_frate

def computeAngularVelocityTuningCurves(spikes, angle, ep, nb_bins = 61, bin_size = 10000, norm=True):
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2             = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    time_bins        = np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
    index             = np.digitize(tmp2.index.values, time_bins)
    tmp3             = tmp2.groupby(index).mean()
    tmp3.index         = time_bins[np.unique(index)-1]+bin_size/2
    tmp3             = nts.Tsd(tmp3)
    tmp4            = np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
    tmp2             = nts.Tsd(tmp2)
    tmp4            = np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)    
    velocity         = nts.Tsd(t=tmp2.index.values[1:], d = tmp4)
    velocity         = velocity.restrict(ep)    
    bins             = np.linspace(-2*np.pi, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    velo_curves        = pd.DataFrame(index = idx, columns = list(spikes.keys()))

    for k in spikes:
        spks         = spikes[k]
        spks         = spks.restrict(ep)
        speed_spike = velocity.realign(spks)
        spike_count, bin_edges = np.histogram(speed_spike, bins)
        occupancy, _ = np.histogram(velocity.restrict(ep), bins)
        spike_count = spike_count/(occupancy+1)
        velo_curves[k] = spike_count*(1/(bin_size*1e-6))
        # normalizing by firing rate 
        if norm:
            velo_curves[k] = velo_curves[k]/(len(spikes[k].restrict(ep))/ep.tot_length('s'))

    return velo_curves

def computeSpeedTuningCurves(spikes, position, ep, bin_size = 0.1, nb_bins = 20, speed_max = 0.4):
    time_bins     = np.arange(position.index[0], position.index[-1]+bin_size*1e6, bin_size*1e6)
    index         = np.digitize(position.index.values, time_bins)
    tmp         = position.groupby(index).mean()
    tmp.index     = time_bins[np.unique(index)-1]+(bin_size*1e6)/2
    distance    = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
    speed         = nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
    speed         = speed.restrict(ep)
    bins         = np.linspace(0, speed_max, nb_bins)
    idx         = bins[0:-1]+np.diff(bins)/2
    speed_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
    for k in spikes:
        spks     = spikes[k]
        spks     = spks.restrict(ep)
        speed_spike = speed.realign(spks)
        spike_count, bin_edges = np.histogram(speed_spike, bins)
        occupancy, _ = np.histogram(speed, bins)
        spike_count = spike_count/(occupancy+1)
        speed_curves[k] = spike_count/bin_size

    return speed_curves

def splitWake(ep):
    if len(ep) != 1:
        print('Cant split wake in 2')
        sys.exit()
    tmp = np.zeros((2,2))
    tmp[0,0] = ep.values[0,0]
    tmp[1,1] = ep.values[0,1]
    tmp[0,1] = tmp[1,0] = ep.values[0,0] + np.diff(ep.values[0])/2
    
    return nts.IntervalSet(start = tmp[:,0], end = tmp[:,1])

def loadMeanWaveforms(path):
    """
    load waveforms
    quick and dirty    
    """
    import scipy.io
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()    
    new_path = os.path.join(path, 'Analysis/')
    if os.path.exists(new_path):
        new_path    = os.path.join(path, 'Analysis/')
        files        = os.listdir(new_path)
        if 'SpikeWaveF.mat' in files:            
            # data = scipy.io.loadmat(path+'/Analysis/SpikeWaveF.mat')
            # meanWaveF = data['meanWaveF'][0]
            # maxIx = data['maxIx'][0]
            # generalinfo     = scipy.io.loadmat(path+'/Analysis/GeneralInfo.mat')
            # shankStructure     = loadShankStructure(generalinfo)
            # spikes,shank = loadSpikeData(path+'/Analysis/SpikeData.mat', shankStructure['thalamus'])    
            # index_neurons = [path.split("/")[-1]+"_"+str(n) for n in spikes.keys()]
            # for i, n in zip(list(spikes.keys()), index_neurons):    
            #     to_return[n] = meanWaveF[i][maxIx[i]-1]            
            print("to test matlab")
            return
        elif "MeanWaveForms.h5" in files and "MaxWaveForms.h5" in files:
            meanwavef = pd.read_hdf(os.path.join(new_path, 'MeanWaveForms.h5'))
            maxch = pd.read_hdf(os.path.join(new_path, 'MaxWaveForms.h5'))
            return meanwavef, maxch

    # Creating /Analysis/ Folder here if not already present
    if not os.path.exists(new_path): os.makedirs(new_path)
    files = os.listdir(path)
    clu_files     = np.sort([f for f in files if 'clu' in f and f[0] != '.'])    
    spk_files      = np.sort([f for f in files if 'spk' in f and f[0] != '.'])
    clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2         = np.sort([int(f.split(".")[-1]) for f in spk_files])
    if len(clu_files) != len(spk_files) or not (clu1 == clu2).any():
        print("Not the same number of clu and res files in "+path+"; Exiting ...")
        sys.exit()    

    # XML INFO
    n_channels, fs, shank_to_channel     = loadXML(path)
    from xml.dom import minidom    
    xmlfile = os.path.join(path, [f for f in files if f.endswith('.xml')][0])
    xmldoc         = minidom.parse(xmlfile)
    nSamples     = int(xmldoc.getElementsByTagName('nSamples')[0].firstChild.data) # assuming constant nSamples

    import xml.etree.ElementTree as ET
    root = ET.parse(xmlfile).getroot()


    count = 0
    meanwavef = []
    maxch = []
    for i, s in zip(range(len(clu_files)),clu1):
        clu = np.genfromtxt(os.path.join(path,clu_files[i]),dtype=np.int32)[1:]
        mwf = []
        mch = []        
        if np.max(clu)>1:
            # load waveforms
            file = os.path.join(path, spk_files[i])
            f = open(file, 'rb')
            startoffile = f.seek(0, 0)
            endoffile = f.seek(0, 2)
            bytes_size = 2
            n_samples = int((endoffile-startoffile)/bytes_size)
            f.close()            
            n_channel = len(root.findall('spikeDetection/channelGroups/group')[s-1].findall('channels')[0])
            data = np.fromfile(open(file, 'rb'), np.int16)
            data = data.reshape(len(clu),nSamples,n_channel)
            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp>1]
            idx_col = np.arange(count, count+len(idx_clu))            
            for j,k in zip(idx_clu, idx_col):
                meanw = data[clu==j].mean(0)
                ch = np.argmax(np.max(np.abs(meanw), 0))
                mwf.append(meanw.flatten())
                mch.append(ch)
            mwf = pd.DataFrame(np.array(mwf).T)
            mwf.columns = pd.Index(idx_col)
            mch = pd.Series(index = idx_col, data = mch)
            count += len(idx_clu)
            meanwavef.append(mwf)
            maxch.append(mch)

    meanwavef = pd.concat(meanwavef, 1)
    maxch = pd.concat(maxch)    
    meanwavef.to_hdf(os.path.join(new_path, 'MeanWaveForms.h5'), key='waveforms', mode='w')
    maxch.to_hdf(os.path.join(new_path, 'MaxWaveForms.h5'), key='channel', mode='w')
    return meanwavef, maxch

def computeSpatialInfo(tc, angle, ep):
    nb_bins = tc.shape[0]+1
    bins = np.linspace(0, 2*np.pi, nb_bins)
    angle = angle.restrict(ep)
    # Smoothing the angle here
    tmp = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2 = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    angle = nts.Tsd(tmp2%(2*np.pi))
    pf = tc.values
    occupancy, _     = np.histogram(angle, bins)
    occ = np.atleast_2d(occupancy/occupancy.sum()).T
    f = np.sum(pf * occ, 0)
    pf = pf / f
    SI = np.sum(occ * pf * np.log2(pf), 0)
    SI = pd.DataFrame(index = tc.columns, columns = ['SI'], data = SI)
    return SI

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
def refineSleepFromAccel(acceleration, sleep_ep):
	vl = acceleration[0].restrict(sleep_ep)
	vl = vl.as_series().diff().abs().dropna()	
	a, _ = scipy.signal.find_peaks(vl, 0.025)
	peaks = nts.Tsd(vl.iloc[a])
	duration = np.diff(peaks.as_units('s').index.values)
	interval = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])

	newsleep_ep = interval.iloc[duration>15.0]
	newsleep_ep = newsleep_ep.reset_index(drop=True)
	newsleep_ep = newsleep_ep.merge_close_intervals(100000, time_units ='us')

	newsleep_ep	= sleep_ep.intersect(newsleep_ep)

	return newsleep_ep

def downsample(tsd, up, down):
	import scipy.signal
	import neuroseries as nts
	dtsd = scipy.signal.resample_poly(tsd.values, up, down)
	dt = tsd.as_units('s').index.values[np.arange(0, tsd.shape[0], down)]
	if len(tsd.shape) == 1:		
		return nts.Tsd(dt, dtsd, time_units = 's')
	elif len(tsd.shape) == 2:
		return nts.TsdFrame(dt, dtsd, time_units = 's', columns = list(tsd.columns))
    
def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import lfilter
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def writeNeuroscopeEvents(path, ep, name):
	f = open(path, 'w')
	for i in range(len(ep)):
		f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
		f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
	f.close()		
	return