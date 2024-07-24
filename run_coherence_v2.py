from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--date', type=str, help='YYYY-MM-DD')
parser.add_argument('--ifo', type=str, help='L1 or H1')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory to save data')
args = parser.parse_args()

t1 = args.date
ifo = args.ifo
savedir = args.savedir

date1 = datetime.strptime(t1, '%Y-%m-%d')
date2 = date1 + timedelta(days=1)
date2 = date2.strftime('%Y-%m-%d')


segs_ = get_observing_segs(date1, date2)
times_segs = get_times(seglist=segs_)

channel_path = 'channel_files/{}/'.format(ifo)

df_all_chans = pd.read_csv(channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel'])

df_chan1 = df_all_chans.iloc[:100]
df_chan2 = df_all_chans.iloc[100:200]
df_chan3 = df_all_chans.iloc[200:300]
df_chan4 = df_all_chans.iloc[300:400]
df_chan5 = df_all_chans.iloc[400:500]
df_chan6 = df_all_chans.iloc[500:600]
df_chan7 = df_all_chans.iloc[600:700]
df_chan8 = df_all_chans.iloc[700:]

time_ = random.choice(times_segs)

print("Time is {}".format(time_))

ht_data = get_strain_data(time_, time_ + 900, ifo=ifo)
print("Got h(t) data")

def give_group(a):
    group = a.split('_')[1]
    
    return group

def get_coherence_chan1(channel_list=df_chan1['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    
    
    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return

def get_coherence_chan2(channel_list=df_chan2['channel'], first_chan_index=0,
                     ifo=ifo, times=time_, strain_data=ht_data):
        
    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    channel_list = channel_list.iloc[first_chan_index:]
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return


def get_coherence_chan3(channel_list=df_chan3['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return 

def get_coherence_chan4(channel_list=df_chan4['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return 

def get_coherence_chan5(channel_list=df_chan5['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return

def get_coherence_chan6(channel_list=df_chan6['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return

def get_coherence_chan7(channel_list=df_chan7['channel'], ifo=ifo, times=time_, strain_data=ht_data):


    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return

def get_coherence_chan8(channel_list=df_chan8['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return



p1 = multiprocessing.Process(target = get_coherence_chan1)
p2 = multiprocessing.Process(target = get_coherence_chan2)
p3 = multiprocessing.Process(target = get_coherence_chan3)
p4 = multiprocessing.Process(target = get_coherence_chan4)
p5 = multiprocessing.Process(target = get_coherence_chan5)
p6 = multiprocessing.Process(target = get_coherence_chan6)
p7 = multiprocessing.Process(target = get_coherence_chan7)
p8 = multiprocessing.Process(target = get_coherence_chan8)


import time
tic = time.time()

p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()


p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()

tac = time.time()
print(tac - tic)


#file_path = '/home/siddharth.soni/O4/coherence_study/coherence_monitor/data/{}/'.format(int(time_))
file_path = os.path.join(savedir, '{}'.format(int(time_)), '')

vals = get_max_corr(file_path, save=True)
vals['group'] = vals['channel'].apply(give_group)

import plotly.express as px
import plotly
fig = px.scatter(vals, x="frequency", y="max_correlation", 
                  hover_data=['channel'], color= "group", labels={"max_correlation": "Max Correlation",
                                                                 "frequency":"Frequency [Hz]"})
fig.update_layout(
    title=dict(text="Max Coherence during {} -- {}".format(str(time_), str(time_ + 900)), font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple")))


plotly.offline.plot(fig, filename = 'plots/scatter_coh_{}.png'.format(int(time_)))
