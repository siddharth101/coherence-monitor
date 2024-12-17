"""Script to run Daily coherence monitor"""

import os
import random
import argparse
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
import logging
from gwpy.timeseries import TimeSeries
import sys
sys.path.append('/home/siddharth.soni/src/coherence-monitor/')
from utils import (
    get_observing_segs,
    get_times,
    calc_coherence,
    run_coherence,
    get_max_corr,
    get_frame_files,
    get_strain_data,
    get_unsafe_channels,
    generate_plots
)


GWDATAFIND_SERVER='datafind.ldas.cit:80'

import os

__author__ = 'Siddharth Soni <siddharth.soni@ligo.org>'


from gwpy.time import tconvert, from_gps, to_gps
from datetime import datetime, timedelta

now  = from_gps(tconvert(gpsordate='now'))



date = now.strftime('%Y-%m-%d')
#date = '2024-11-20'


date_ = datetime.strptime(date, '%Y-%m-%d')
date_yesterday = date_ + timedelta(days=-1)
date_yesterday_ = date_yesterday.strftime('%Y-%m-%d')
date_tomorrow = date_ + timedelta(days=1)


gps_yesterday = to_gps(date_yesterday).gpsSeconds
gps_today = to_gps(date_).gpsSeconds

t1 = gps_yesterday + 8*3600
t2 = gps_yesterday + 16*3600
t3 = gps_today


ifo = 'L1'
dur = 1024
savedir = f'/home/siddharth.soni/public_html/coherence_monitor/{ifo}/'

coh_thresh = 0.1 


if not os.path.exists(savedir):
    os.makedirs(savedir)

savedir_path = os.path.join(savedir, date_yesterday_, 'data', '')
logfilepath = os.path.join(savedir, date_yesterday_, '')

if not os.path.exists(savedir_path):
    os.makedirs(savedir_path)
    
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(f"{logfilepath}log.log"),  # Save all logs here
        logging.StreamHandler()  # Prints logs to console as well
    ]
)



segs_ = get_observing_segs(t2, t3, ifo)
if segs_:
    times_segs = get_times(seglist=segs_, duration=3600)
    logging.info("Got the segments")
    logging.info(f"The coherence monitor will run for each of the times in {times_segs}")
else:
    logging.info("No Observing segments")

    
    
    
    
def give_group(a):
    group = a.split('_')[1]
    return group


def get_coherence_chan(channel_list, gpstime, ifo, strain_data, dur):
    #GWDATAFIND_SERVER = 'datafind.ldas.cit:80'
    files_ = get_frame_files(gpstime, gpstime + dur, ifo=ifo)
    # logging.info("Got {} files".format(len(files_)))
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

    logging.info(f"Running the coherence monitor on {date_yesterday_} for {ifo}")
    logging.info(f"Calculating coherence for {ifo} between {t2} and {t3}")

    channel_path = '/home/siddharth.soni/src/coherence-monitor/channel_files/{}/'.format(ifo)

    df_safe = pd.read_csv(channel_path + '{}_safe_channels.csv'.format(ifo))

    # df_all_chans = pd.read_csv(channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel'])
    # df_unsafe_chans = get_unsafe_channels(ifo)

    logging.info("Total safe auxiliary channels: {}".format(len(df_safe)))

    #df_all_chans = df_all_chans[~df_all_chans['channel'].isin(df_unsafe_chans['channel'])]

    #logging.info("Total auxiliary channels after removing unsafe channels: {}".format(len(df_all_chans)))
    
    try:
        import time

        for i in times_segs:
            tic = time.time()
            time_ = i + 300 #adding 300 seconds in case the start time is very close to relock time
            logging.info("Time is {}".format(time_))

            ht_data = get_strain_data(time_, time_ + dur, ifo=ifo)

            logging.info("Got h(t) data between {} and {}".format(time_, time_ + dur))
            logging.info(ht_data.duration)

            run_process(df_safe, time=time_, ifo=ifo, strain_data=ht_data, dur=dur)

            tac = time.time()
            logging.info(tac - tic)

        logging.info("Analysis done now, making plots")
        generate_plots(date_yesterday_, ifo=ifo)

#         path_outdir = os.path.join(savedir, date, str(gps_today), 'plots', '')
#         if not os.path.exists(path_outdir):
#             os.makedirs(path_outdir)

#         #dirs_path = savedir_path #os.path.join(savedir, date1, 'data', '')
#         logging.info(savedir_path)

#         for filepath in os.listdir(savedir_path):
#             path_ = os.path.join(savedir_path, filepath, '')
#             logging.info(path_)
#             t = path_.split('/')[-2]
#             savedir_plots = os.path.join(path_outdir, t, '')
#             os.makedirs(savedir_plots, exist_ok=True)
#             plot_max_corr_chan(path=path_, ifo=ifo, fft=10, savedir=savedir_plots)
    except Exception as e:
        logging.error(f"An error occurred: {e}")   
        
else:
    pass
    #logging.info("No Observing segments")
