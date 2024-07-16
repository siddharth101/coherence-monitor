from gwpy.time import to_gps, from_gps
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import gwdatafind
import numpy as np
import pandas as pd
import os
import glob

def get_strain_data(starttime, endtime, ifo='L1'):
    
    ht = TimeSeries.get('{}:GDS-CALIB_STRAIN'.format(ifo), starttime, endtime)
    
    return ht
    

def get_frame_files(starttime, endtime, ifo='L1'):
    
    site_ = ifo[0]
    files = gwdatafind.find_urls(f'{site_}',f'{ifo}_R',starttime, endtime)
    files = sorted(files)
    
    return files 


def get_observing_segs(t1, t2, ifo='L1'):
    
    tstart = to_gps(t1)
    tend = to_gps(t2)
    
    segs = DataQualityFlag.query(f'{ifo}:DMT-ANALYSIS_READY:1', t1, t2)
    
    seg_list = SegmentList()
    for i in segs.active:
        if i.end - i.start > 3600:
            seg_list.append(i)
            
    print("Got the segments")
    return seg_list


def get_times(seglist, duration=3600):
    
    times = [np.arange(i.start, i.end,  3600) for i in seglist]
    
    try:
        times_ = [item for sublist in times for item in sublist]
    except:
        times_ = times
    
    return times_



def calc_coherence(channel2, frame_file, start_time, end_time, fft, overlap, strain_data, channel1=None):
    
    t1 = to_gps(start_time)
    t2 = to_gps(end_time)
    
    ts2 = TimeSeries.read(frame_file, channel=channel2, start=t1, end=t2)
    ts1 = strain_data #TimeSeries.fetch(channel1, t1, t2)
    
    ts1 = ts1.resample(ts2.sample_rate)
    
    coh = ts1.coherence(ts2, fftlength=fft, overlap=overlap)
    
    return coh


def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, ifo='L1'):
    
    t1, t2 = to_gps(starttime), to_gps(endtime)
    
    savedir = os.path.join(savedir, '{}'.format(t1), '')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    h_t = '{}:GDS-CALIB_STRAIN'.format(ifo)
    for i in channel_list:
        print(f"Calculating coherence between DARM and {i}")
        coh = calc_coherence(strain_data=strain_data,channel1=None,
                             channel2= i, frame_file = frame_files, 
                             start_time = t1, end_time = t2, fft=10, overlap=6)
        coh.write(savedir + i.replace(':', '_').replace('-','_')+'_{}_{}.csv'.format(t1, t2))
        
    return
                  

def get_max_corr(output_dir, save=False):
    files = glob.glob(output_dir + '*.csv')
    vals = []
    df_vals_ = pd.DataFrame()
    for i in files:
        chan_name = i.split('/')[-1].split('DQ')[0] + 'DQ'
        #print(chan_name)
        fs = FrequencySeries.read(i)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(1/(n_diff)), int(200/(n_diff))
        
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value
        
        
        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals =  pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency'] )
            df_vals_ = df_vals[df_vals['max_correlation']>0]
        else:
            vals.append(-1)
            
    
        
            
        
    return df_vals_



