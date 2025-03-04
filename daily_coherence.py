"""Script to run Daily coherence monitor"""

import os
import random
import argparse
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
from gwpy.timeseries import TimeSeries
import logging
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




parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--date', type=str, help='YYYY-MM-DD')
parser.add_argument('--ifo', type=str, help='L1 or H1')
parser.add_argument('--dur', type=float, default=1024.0, help='duration of data in secs')
parser.add_argument('--cohthresh', type=float, default=0.1, help='coherence threshold for channel files')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory to save data')
args = parser.parse_args()

t1 = args.date
ifo = args.ifo
dur = args.dur
coh_thresh = args.cohthresh
savedir = args.savedir

if not os.path.exists(savedir):
    os.makedirs(savedir)

date1 = datetime.strptime(t1, '%Y-%m-%d')
date2 = date1 + timedelta(days=1)
date2 = date2.strftime('%Y-%m-%d')

savedir_path = os.path.join(savedir, t1, 'data', '')

if not os.path.exists(savedir_path):
    os.makedirs(savedir_path)

segs_ = get_observing_segs(date1, date2, ifo)
times_segs = get_times(seglist=segs_, duration=3600)

logfilepath = os.path.join(savedir, t1, '')

logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(f"{logfilepath}log.log"),  # Save all logs here
        logging.StreamHandler()  # Prints logs to console as well
    ]
)



channel_path = 'channel_files/{}/'.format(ifo)

df_all_chans = pd.read_csv(channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel'])
df_unsafe_chans = get_unsafe_channels(ifo)

# logging.info(f"Running the daily coherence monitor for {ifo} on {date1}")
# logging.info("Total auxiliary channels: {}".format(len(df_all_chans)))

df_all_chans = df_all_chans[~df_all_chans['channel'].isin(df_unsafe_chans['channel'])]

#logging.info("Total auxiliary channels after removing unsafe channels: {}".format(len(df_all_chans)))


def give_group(a):
    group = a.split('_')[1]
    return group


def get_coherence_chan(channel_list, gpstime, ifo, strain_data, dur):
    files_ = get_frame_files(gpstime, gpstime + dur, ifo=ifo)
    #print("Got {} files".format(len(files_)))
    run_coherence(
        channel_list=channel_list,
        frame_files=files_,
        starttime=gpstime,
        endtime=gpstime + dur,
        ifo=ifo,
        strain_data=strain_data,
        savedir=savedir_path,
        coh_thresh = coh_thresh
    )
    return


def run_process(channel_df, time, ifo, strain_data, dur):
    processes = [
            multiprocessing.Process(
                target=get_coherence_chan,
                args=(channel_df.iloc[i:i + 60]['channel'], time, ifo, strain_data, dur),
            )
            for i in range(0, 900, 60)
        ]

    [p.start() for p in processes]
    [p.join() for p in processes]

    return


if times_segs:
    
    try:
    
#         import time

#         for i in times_segs:
# #             tic = time.time()
#             time_ = i
#             logging.info("Time is {}".format(time_))

#             ht_data = get_strain_data(time_, time_ + dur, ifo=ifo)

#             logging.info("Got h(t) data between {} and {}".format(time_, time_ + dur))
#             #print(ht_data.duration)

#             run_process(df_all_chans, time=time_, ifo=ifo, strain_data=ht_data, dur=dur)

#             tac = time.time()
#             logging.info(f"Took {tac - tic} secs to run this batch")

        logging.info("Analysis done now, making plots")

        public_html = '/home/siddharth.soni/public_html/coherence_monitor/{}'.format(ifo)
        path_outdir = os.path.join(public_html, args.date, 'plots', '')
        if not os.path.exists(path_outdir):
            os.makedirs(path_outdir)

        dirs_path = savedir_path #os.path.join(args.savedir, args.date, 'data', '')
        print(dirs_path)

        for filepath in os.listdir(dirs_path):
            path_ = os.path.join(dirs_path, filepath, '')
            print(path_)
            savedir_plots = os.path.join(path_outdir)
            if not os.path.exists(savedir_plots):
                os.makedirs(savedir_plots)
            plot_max_corr_chan(path=path_, ifo=args.ifo, fft=10, savedir=savedir_plots)
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")       
else:
    logging.info("No Observing segments found")
