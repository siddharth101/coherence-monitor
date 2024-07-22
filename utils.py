from gwpy.time import to_gps, from_gps
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import gwdatafind
import numpy as np
import pandas as pd
import os
import glob
import plotly.express as px
import plotly

def give_group_v2(a):
    group = a.split(':')[1].split('_')[0]
    
    return group

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

def get_max_corr_band(output_dir,flow=10, fhigh=20, save=False):
    files = glob.glob(output_dir + '*.csv')
    vals = []
    df_vals_ = pd.DataFrame()
    for i in files:
        chan_name = i.split('/')[-1].split('DQ')[0] + 'DQ'
        #print(chan_name)
        fs = FrequencySeries.read(i)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(flow/(n_diff)), int(fhigh/(n_diff))
        
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

def combine_csv(dir_path):
    path = dir_path # use your path
    all_files = glob.glob(path + "*.csv")
    
    all_files = [i for i in all_files if not i.startswith(path+'L1_OMC_DCPD_SUM_OUT')]
    
    
    
    li = []
    
    fns = []
    for filename in all_files:
        fn = filename.split('/')[-1].split('_14')[0] + '_freq'
        fn_val = filename.split('/')[-1].split('_14')[0] + '_corr'
        fns.append(fn)
        fns.append(fn_val)
        df = pd.read_csv(filename, index_col=None, header=None)
        li.append(df)
    frame = pd.concat(li, axis=1, ignore_index=True)
    frame.columns = fns
    return frame

def find_max_corr_channel(path,fft=10,ifo='L1'):
    frame_  = combine_csv(path)
    
    #frame_ = frame.iloc[:]
    
    max_vals = []
    for i in range(len(frame_)):
        max_val_ = frame_.iloc[i,1::2].sort_values(ascending=False)
        chan_names =  max_val_.index[0:2]
        chan_names = [i.replace('_corr', '').replace(f'{ifo}_', f'{ifo}:') for i in chan_names]
        
        max_corr_val = [max_val_.iloc[0], max_val_.iloc[1]]
        
        
        max_vals.append(((i)/fft,chan_names[0], max_corr_val[0],chan_names[1], max_corr_val[1]))
        
        
    df = pd.DataFrame(max_vals, columns=['frequency', 'channel1', 'corr1', 'channel2', 'corr2'])
    return df
        


def plot_max_corr_chan(path, fft, ifo):
    
    time_ = int(path.split('/')[-2])
    vals = find_max_corr_channel(path=path, fft=fft, ifo=ifo)
    print("Got the data, now making plots")
    vals = vals.iloc[:2001] # this would need to incorportate fft information
    vals['group1'] = vals['channel1'].apply(give_group_v2)
    vals['group2'] = vals['channel2'].apply(give_group_v2)


    fig1 = px.scatter(vals, x="frequency", y="corr1", 
                  hover_data=['channel1'], color= "group1", labels={"max_correlation": "Max Correlation",
                                                                 "frequency":"Frequency [Hz]"})
    
    fig2 = px.scatter(vals, x="frequency", y="corr2", 
                  hover_data=['channel2'], color= "group2", labels={"max_correlation": "Max Correlation",
                                                                 "frequency":"Frequency [Hz]"})
    fig1.update_layout(
    title=dict(text="Highest Coherence channel at each frequency during {} -- {}".format(str(time_), str(time_ + 900)), font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple")))
    
    fig2.update_layout(
    title=dict(text="Second highest Coherence channel at each frequency during {} -- {}".format(str(time_), str(time_ + 900)), font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple")))


    plotly.offline.plot(fig1, filename = 'plots/channels_coh_{}_a.png'.format(int(time_)))
    plotly.offline.plot(fig2, filename = 'plots/channels_coh_{}_b.png'.format(int(time_)))
    
    return