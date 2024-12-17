"""Script to run coherence between strain data and auxiliary data for any given time"""

import os
import random
import argparse
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
from gwpy.timeseries import TimeSeries
from utils import (
    get_observing_segs,
    get_times,
    calc_coherence,
    run_coherence,
    get_max_corr,
    get_frame_files,
    get_strain_data,
    get_unsafe_channels,
)


__author__ = 'Siddharth Soni <siddharth.soni@ligo.org>'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--t1', type=float, help='gps time', default=None)
parser.add_argument('--t2', type=float, help='gps time', default=None)
parser.add_argument('--ifo', type=str, help='L1 or H1')
#parser.add_argument('--dur', type=float, default=1024.0, help='duration of data in secs')
parser.add_argument('--cohthresh', type=float, default=0.1, help='coherence threshold for channel files')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory to save data')
args = parser.parse_args()


ifo = args.ifo
start_time = args.t1
end_time = args.t2
savedir = args.savedir
coh_thresh = args.cohthresh

if end_time - start_time < 100:
    end_time = start_time + 100
    print(f"End time as too close to starttime, new endtime is {end_time}")
else:
    pass

segs_ = get_observing_segs(start_time, end_time, ifo=ifo)
times_segs = get_times(seglist=segs_)

channel_path = 'channel_files/{}/'.format(ifo)

df_safe = pd.read_csv(channel_path + '{}_safe_channels.csv'.format(ifo))

# df_all_chans = pd.read_csv(
#     channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel']
# )

# df_unsafe_chans = get_unsafe_channels(ifo)
print("Total safe auxiliary channels: {}".format(len(df_safe)))

# df_all_chans = df_all_chans[~df_all_chans['channel'].isin(df_unsafe_chans['channel'])]

# print("Total auxiliary channels after removing unsafe channels: {}".format(len(df_all_chans)))


ht_data = get_strain_data(start_time, end_time, ifo=ifo)

print("Got h(t) data between {} and {}".format(start_time, end_time))
print(ht_data.duration)


def give_group(a):
    group = a.split('_')[1]
    return group


def get_coherence_chan(channel_list, starttime, endtime, ifo, strain_data):
    files_ = get_frame_files(starttime, endtime, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(
        channel_list=channel_list,
        frame_files=files_,
        starttime=starttime,
        endtime=endtime,
        ifo=ifo,
        strain_data=strain_data,
        savedir=savedir,
        coh_thresh = coh_thresh
    )
    return


def run_process(channel_df, ifo=ifo):
    if ifo=='H1':
        n_chans = 960
    else:
        n_chans = 900
    processes = [
            multiprocessing.Process(
                target=get_coherence_chan,
                args=(channel_df.iloc[i:i + 60]['channel'], start_time,
                      end_time, ifo, ht_data),
            )
            for i in range(0, n_chans, 60)
        ]

    [p.start() for p in processes]
    [p.join() for p in processes]

    return


import time
tic = time.time()

run_process(df_safe)

tac = time.time()
print(tac - tic)
