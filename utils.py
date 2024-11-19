"""Utilities for coherence monitor"""

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

__author__ = 'Siddharth Soni <siddharth.soni@ligo.org>'


def give_group_v2(a):
    group = a.split(':')[1].split('-')[0]
    return group


def get_strain_data(starttime, endtime, ifo='L1'):
    ht = TimeSeries.get(f'{ifo}:GDS-CALIB_STRAIN', starttime, endtime)
    return ht


def get_frame_files(starttime, endtime, ifo, host=None):
    site_ = ifo[0]
    if host:
        files = gwdatafind.find_urls(f'{site_}', f'{ifo}_R', starttime, endtime, host)
    else:
        files = gwdatafind.find_urls(f'{site_}', f'{ifo}_R', starttime, endtime)
        
    return sorted(files)


def get_unsafe_channels(ifo):
    path = f'/home/siddharth.soni/src/coherence-monitor/channel_files/{ifo}/{ifo}_unsafe_channels.csv'
    return pd.read_csv(path)


def get_observing_segs(t1, t2, ifo):
    tstart = to_gps(t1)
    tend = to_gps(t2)
    segs = DataQualityFlag.query(f'{ifo}:DMT-ANALYSIS_READY:1', t1, t2)
    seg_list = SegmentList()

    for seg in segs.active:
        if seg.end - seg.start > 3600:
            seg_list.append(seg)

    # if seg_list:
    #     print("Got the segments")
    return seg_list


def get_times(seglist, duration=3600):
    times = [np.arange(i.start, i.end - duration, duration) for i in seglist]
    return [item for sublist in times for item in sublist]


def calc_coherence(channel2, frame_file=None, start_time, end_time, fft, overlap, strain_data, channel1=None):
    t1 = to_gps(start_time)
    t2 = to_gps(end_time)
    if frame_file:
        ts2 = TimeSeries.read(frame_file, channel=channel2, start=t1, end=t2)
    else:
        ts2 = TimeSeries.fetch(channel2, start=t1, end=t2)
        

    if channel1:
        ts1 = TimeSeries.fetch(channel1, t1, t2)
    else:
        ts1 = strain_data

    ts1 = ts1.resample(ts2.sample_rate)
    coh = ts1.coherence(ts2, fftlength=fft, overlap=overlap)

    for i in np.where(coh.value == np.inf)[0]:
        try:
            coh.value[i] = (coh.value[i - 2] + coh.value[i - 1]) / 2
        except IndexError:
            coh.value[i] = 1e-20

    return coh



import concurrent.futures


def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, coh_thresh, ifo='L1', timeout=30):
    t1, t2 = to_gps(starttime), to_gps(endtime)
    savedir = os.path.join(savedir, f'{t1}', '')

    if not os.path.exists(savedir):
        print(f"Creating the output dir {savedir}")
        os.makedirs(savedir)

    for channel in channel_list:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(calc_coherence, strain_data=strain_data, channel1=None,
                                     channel2=channel, frame_file=frame_files,
                                     start_time=t1, end_time=t2, fft=10, overlap=5)

            try:
                coh = future.result(timeout=timeout)
                coh_ = coh[coh.value > coh_thresh]
                if len(coh_) > 0:
                    coh_.write(f'{savedir}{channel}.csv')

            except concurrent.futures.TimeoutError:
                print(f"Calculation for channel {channel} timed out.")
                continue


def get_max_corr(output_dir, save=False):
    files = glob.glob(f'{output_dir}*.csv')
    vals = []

    for file in files:
        chan_name = file.split('/')[-1].split('DQ')[0] + 'DQ'
        fs = FrequencySeries.read(file)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(1 / n_diff), int(200 / n_diff)
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value

        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals = pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency'])
            df_vals_ = df_vals[df_vals['max_correlation'] > 0]
        else:
            vals.append(-1)

    return df_vals_


def get_max_corr_band(output_dir, flow=10, fhigh=20, save=False):
    files = glob.glob(f'{output_dir}*.csv')
    vals = []

    for file in files:
        chan_name = file.split('/')[-1].split('DQ')[0] + 'DQ'
        fs = FrequencySeries.read(file)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(flow / n_diff), int(fhigh / n_diff)
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value

        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals = pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency'])
            df_vals_ = df_vals[df_vals['max_correlation'] > 0]
        else:
            vals.append(-1)

    return df_vals_


def combine_csv(dir_path, ifo):
    all_files = glob.glob(f'{dir_path}*.csv')
    chan_removes = get_unsafe_channels(ifo=ifo)['channel']

    for j in chan_removes:
        all_files = [i for i in all_files if not i.startswith(f'{dir_path}{j}')]

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


# def find_max_corr_channel(path, fft=10, ifo='L1'):
#     frame_ = combine_csv(path, ifo)
#     max_vals = []

#     for i in range(len(frame_)):
#         max_val_ = frame_.iloc[i, 1::2].sort_values(ascending=False)
#         chan_names = max_val_.index[:2]
#         chan_names = [i.replace('_corr', '').replace('.csv','') for i in chan_names]
#         max_corr_val = [max_val_.iloc[0], max_val_.iloc[1]]
#         max_vals.append((i / fft, chan_names[0], max_corr_val[0], chan_names[1], max_corr_val[1]))

#     df = pd.DataFrame(max_vals, columns=['frequency', 'channel1', 'coh1', 'channel2', 'coh2'])
#     return df


# def plot_max_corr_chan(path, fft, ifo, flow=0, fhigh=200, plot=True, savedir=None):
#     time_ = int(path.split('/')[-2])
#     vals = find_max_corr_channel(path=path, fft=fft, ifo=ifo)
#     print("Got the data, now making plots")
#     vals = vals.iloc[flow * fft:fhigh * fft + 1]
#     vals['group1'] = vals['channel1'].apply(give_group_v2)
#     vals['group2'] = vals['channel2'].apply(give_group_v2)
#     vals = vals[(vals['coh1'] <=1) & (vals['coh2']<=1)]
#     vals.rename(columns={'coh1':'coherence', 'coh2':'Coherence'},  inplace=True)

#     if plot:
#         fig1 = px.scatter(vals, x="frequency", y="coherence", hover_data=['channel1'], color="group1",
#                           labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"})
#         fig1.update_layout(
#             title=dict(text=f"Highest Coherence channel at each frequency during {time_} -- {time_ + 900}",
#                        font=dict(family="Courier New, monospace", size=28, color="RebeccaPurple")))

#         fig2 = px.scatter(vals, x="frequency", y="coherence", hover_data=['channel2'], color="group2",
#                           labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"})
#         fig2.update_layout(
#             title=dict(text=f"Second highest Coherence channel at each frequency during {time_} -- {time_ + 900}",
#                        font=dict(family="Cour ier New, monospace", size=28, color="RebeccaPurple")))

#         plotly.offline.plot(fig1, filename=f'{savedir}/channels_coh_{int(time_)}_a.png')
#         plotly.offline.plot(fig2, filename=f'{savedir}/channels_coh_{int(time_)}_b.png')

#     return vals


def check_channel_coherence(channel, ifo, t1, t2, fft=10, overlap=5):

    files = get_frame_files(starttime=t1, endtime=t2, ifo=ifo)
    ts_aux = TimeSeries.read(files, channel=channel, start=t1, end=t2)
    ts_gds = TimeSeries.fetch('{}:GDS-CALIB_STRAIN'.format(ifo), t1, t2)

    ts_gds = ts_gds.resample(ts_aux.sample_rate)
    coh = ts_gds.coherence(ts_aux, fftlength=fft, overlap=overlap)

    return coh

def create_dataframe(files, threshold=0.6, frequency_threshold=200):
    
    coh_threshold = threshold
    freq_threshold = frequency_threshold
    
    df_list = []
    for file in files:
            df = pd.read_csv(file, header=None, names=['frequency', 'coherence'])
            df = df[(df['coherence']>coh_threshold) & 
                    (df['frequency']<freq_threshold) ]# & ((df['frequency']>61) | (df['frequency']<59))]
            if len(df)!=0:
                chan_name = file.split('/')[-1].split('.csv')[0]
                df['channel'] = [chan_name]*len(df)
                df_list.append(df)
            
    if df_list:
        frame = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        frame = pd.DataFrame()
    
    return frame

def freq_masks(df):
    freq_mask = ((df.frequency >= 59.8) & (df.frequency <= 60.2)) | ((df.frequency >= 119.8) & (df.frequency <= 120.2)) \
    | ((df.frequency >= 179.8) & (df.frequency <= 180.2))

    return df[~freq_mask]

def coherence_above(ifo, date, path=None):
    
    gpstime = to_gps(date)
    if path is None:
        path = f'/home/siddharth.soni/public_html/coherence_monitor/{ifo}/{date}/{gpstime}/data/'
    else:
        path = path
    times = [os.path.join(path, i, '') for i in os.listdir(path)]
    
    if not times:
        print("No data found for this date")
    for time in times:
        gpstime_ = time.split('/')[-2]
        #print(gpstime_)
        print(f"Running for {time}")
        files = glob.glob(time + '*.csv')
        df = create_dataframe(files)
        df = freq_masks(df)
        df['time'] = gpstime_
        df.reset_index(drop=True, inplace=True)
        
        #print(len(df))
        if len(df)!=0:
            maxpath = path.replace('data/','')
            path_ = os.path.join(maxpath, 'max_coherence','')
            os.makedirs(path_, exist_ok=True)
                #print(path_)
            df.to_csv(path_ + f'coherence_above_{gpstime_}.csv', index=None)
        
    return
            
    
    
    
def combine_data_files(path):
    
    files = glob.glob(path + '*.csv')
    
    nfiles = len(files)
    li = []
    for i in range(nfiles):
        file = files[i]
        channame = file.split('.csv')[-2].split('/')[-1]
        df = pd.read_csv(file, names = ['freq', 'value'])
        df.freq = df.freq.round(1)
        df.value = df.value.round(2)
        df['channel'] = channame
        if len(df)>0:
            li.append(df)
            
        frame = pd.concat(li, axis=0, ignore_index=True)
        
    return frame

def get_max_coherence(dataframe, min_freq=0.0, max_freq=100.0, fft=10):
    
    res = 1/fft
    frequencies = np.arange(min_freq,max_freq+res,res)
    df = dataframe
    
    max_val = []
    limaxdf = []
    for frequency in frequencies:
        frequency = round(frequency, 1)
        try:
            dfmax = df[df['freq']==frequency].sort_values(by='value', ascending=False).iloc[0]
            limaxdf.append(dfmax)
        except:
            pass
        
        
    frame_limaxdf = pd.concat(limaxdf, axis=1, ignore_index=True).T
    
    return frame_limaxdf


def generate_plots(date, ifo):
    
    if ifo == 'L1':
        pathifo = '/home/siddharth.soni/public_html/coherence_monitor/L1/'
    else:
        pathifo = '/home/siddharth.soni/public_html/coherence_monitor/H1/'
        
    gpstime = to_gps(date).gpsSeconds
    
    folder_path = os.path.join(pathifo, date, str(gpstime), 'data','')
    gps_folders = os.listdir(folder_path)
    
    for folder in gps_folders:
        gps_folder_path = os.path.join(folder_path, folder, '' )
        #print(gps_folder_path)
        fr = combine_data_files(gps_folder_path)
        frmax = get_max_coherence(fr, min_freq=0.0, max_freq=200.0, fft=10)
        vals = frmax
        vals['group'] = vals['channel'].apply(give_group_v2)
        vals.rename(columns={'value':'Coherence', 'freq':'Frequency'},  inplace=True)
        
        plotdir = os.path.join(pathifo, date, str(gpstime), 'plots', folder,'')
        os.makedirs(plotdir, exist_ok=True)
        print(plotdir)
        fig1 = px.scatter(vals, x="Frequency", y="Coherence", hover_data=['channel'], color="group",
                          labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"})
        
        fig1.update_layout(title=dict(text=f"{ifo}: Highest Coherence channel at each frequency during {folder} -- {str(int(folder) + 1024)}",
                       font=dict(family="Courier New, monospace", size=22, color="RebeccaPurple")))
        plotly.offline.plot(fig1, filename=f'{plotdir}channels_coh_{int(folder)}.png')
        
        
    
    return