"""Script to run Daily coherence monitor"""

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
    plot_max_corr_chan,
)


__author__ = 'Siddharth Soni <siddharth.soni@ligo.org>'


from gwpy.time import tconvert, from_gps, to_gps
from datetime import datetime, timedelta

now  = from_gps(tconvert(gpsordate='now'))

date = now.strftime('%Y-%m-%d')
date_ = datetime.strptime(date, '%Y-%m-%d')
date_yesterday = date_ - timedelta(1)

date_ = date_.strftime('%Y-%m-%d')
date_yesterday = date_yesterday.strftime('%Y-%m-%d')


date1 = date_yesterday
date2 = date_
ifo = 'H1'
dur = 1024
savedir = '/home/siddharth.soni/public_html/coherence_monitor/H1/'
coh_thresh = 0.1 

# print("Calculating coherence for {} on {}".format(ifo, date1))

# if not os.path.exists(savedir):
#     os.makedirs(savedir)

# savedir_path = os.path.join(savedir, date1, 'data', '')

# if not os.path.exists(savedir_path):
#     os.makedirs(savedir_path)

# segs_ = get_observing_segs(date1, date2, ifo)
# times_segs = get_times(seglist=segs_, duration=3600)

# channel_path = 'channel_files/{}/'.format(ifo)

# df_all_chans = pd.read_csv(channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel'])
# df_unsafe_chans = get_unsafe_channels(ifo)

# print("Total auxiliary channels: {}".format(len(df_all_chans)))

# df_all_chans = df_all_chans[~df_all_chans['channel'].isin(df_unsafe_chans['channel'])]

# print("Total auxiliary channels after removing unsafe channels: {}".format(len(df_all_chans)))


# def give_group(a):
#     group = a.split('_')[1]
#     return group


# def get_coherence_chan(channel_list, gpstime, ifo, strain_data, dur):
#     files_ = get_frame_files(gpstime, gpstime + dur, ifo=ifo)
#     print("Got {} files".format(len(files_)))
#     run_coherence(
#         channel_list=channel_list,
#         frame_files=files_,
#         starttime=gpstime,
#         endtime=gpstime + dur,
#         ifo=ifo,
#         strain_data=strain_data,
#         savedir=savedir_path,
#         coh_thresh = coh_thresh
#     )
#     return


# def run_process(channel_df, time, ifo, strain_data, dur):
#     processes = [
#             multiprocessing.Process(
#                 target=get_coherence_chan,
#                 args=(channel_df.iloc[i:i + 60]['channel'], time, ifo, strain_data, dur),
#             )
#             for i in range(0, 900, 60)
#         ]

#     [p.start() for p in processes]
#     [p.join() for p in processes]

#     return


# import time

# for i in times_segs:
#     tic = time.time()
#     time_ = i
#     print("Time is {}".format(time_))

#     ht_data = get_strain_data(time_, time_ + dur, ifo=ifo)

#     print("Got h(t) data between {} and {}".format(time_, time_ + dur))
#     print(ht_data.duration)

#     run_process(df_all_chans, time=time_, ifo=ifo, strain_data=ht_data, dur=dur)

#     tac = time.time()
#     print(tac - tic)

# print("Analysis done now, making plots")

# public_html = '/home/siddharth.soni/public_html/coherence_monitor/{}'.format(ifo)
# path_outdir = os.path.join(public_html, date1, 'plots', '')
# if not os.path.exists(path_outdir):
#     os.makedirs(path_outdir)

# dirs_path = savedir_path #os.path.join(savedir, date1, 'data', '')
# print(dirs_path)

# for filepath in os.listdir(dirs_path):
#     path_ = os.path.join(dirs_path, filepath, '')
#     print(path_)
#     savedir_plots = os.path.join(path_outdir)
#     if not os.path.exists(savedir_plots):
#         os.makedirs(savedir_plots)
#     plot_max_corr_chan(path=path_, ifo=ifo, fft=10, savedir=savedir_plots)
