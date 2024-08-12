from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--date', type=str, help='YYYY-MM-DD', default=None)
parser.add_argument('--time', type=float, help='gps time', default=None)
parser.add_argument('--ifo', type=str, help='L1 or H1')
parser.add_argument('--dur', type=float, default=1024.0, help='duration of data in secs')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory to save data')
args = parser.parse_args()

if args.date is not None:
    t1 = args.date
    date1 = datetime.strptime(t1, '%Y-%m-%d')
    date2 = date1 + timedelta(days=1)
    date2 = date2.strftime('%Y-%m-%d')
elif args.time is not None:
    t1 = args.time
    date1 = t1
    date2 = t1+86400
else: 
    raise Exception("Either date or GPS time needs to be defined!")

ifo = args.ifo
dur = args.dur
savedir = args.savedir

segs_ = get_observing_segs(date1, date2)
times_segs = get_times(seglist=segs_)

channel_path = 'channel_files/{}/'.format(ifo)

df_all_chans = pd.read_csv(channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel'])

if args.date is not None:
    time_ = random.choice(times_segs)
if args.time is not None:
    time_ = t1

print("Time is {}".format(time_))

ht_data = get_strain_data(time_, time_ + dur, ifo=ifo)

print("Got h(t) data between {} and {}".format(time_, time_+dur))
print(ht_data.duration)

def give_group(a):
    group = a.split('_')[1]
    
    return group

def get_coherence_chan(channel_list, ifo=ifo, gpstime=time_, strain_data=ht_data, dur=dur):
   
    files_ = get_frame_files(gpstime, gpstime + dur, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=gpstime, 
                  endtime=gpstime+dur, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return

def run_process(channel_df):
    
    
    processes = [multiprocessing.Process(target = get_coherence_chan, 
                             args=(channel_df.iloc[i:i+50]['channel'], 
                                   ifo, time_, ht_data, dur)) for i in range(0,900,50)]
    
    [i.start() for i in processes]
    [i.join() for i in processes]
    
    return


import time
tic = time.time()

run_process(df_all_chans)

tac = time.time()
print(tac - tic)


