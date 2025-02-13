"""Script to run Daily coherence monitor"""

import os
import random
import argparse
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
import logging
from gwpy.timeseries import TimeSeries
#from calendar import generate_calendar_with_links_for_years
import sys
import shutil
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
#date = '2025-02-07'

date_ = datetime.strptime(date, '%Y-%m-%d')
date_tomorrow = date_ + timedelta(days=1)


gps_today = to_gps(date).gpsSeconds
gps_tomorrow = to_gps(date_tomorrow).gpsSeconds

t1 = gps_today + 8*3600
t2 = gps_today + 16*3600
t3 = gps_tomorrow


ifo = 'L1'
dur = 1024
savedir = f'/home/siddharth.soni/public_html/coherence_monitor/{ifo}/'

coh_thresh = 0.1 

if not os.path.exists(savedir):
    os.makedirs(savedir)

savedir_path = os.path.join(savedir, date, 'data', '')
logfilepath = os.path.join(savedir, date, '')

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



segs_ = get_observing_segs(gps_today, t1, ifo)
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

    logging.info(f"Running the coherence monitor on {date} for {ifo}")
    logging.info(f"Calculating coherence for {ifo} between {gps_today} and {t1}")

    channel_path = '/home/siddharth.soni/src/coherence-monitor/channel_files/{}/'.format(ifo)

    df_safe = pd.read_csv(channel_path + '{}_safe_channels.csv'.format(ifo))

    logging.info("Total safe auxiliary channels: {}".format(len(df_safe)))
    
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
        generate_plots(date, ifo=ifo)
        try:
            path_plots = os.path.join(savedir, date, 'plots', '')
            plot_path = f'/home/siddharth.soni/public_html/coherence_monitor/plots/{ifo}/{date}'
            os.makedirs(plot_path, exist_ok=True)
            shutil.copytree(path_plots,plot_path, 
                            dirs_exist_ok=True)
        except FileNotFoundError:
            pass
        try:
            from utils import generate_calendar_with_links_for_years
            base_dir = "/home/siddharth.soni/public_html/coherence_monitor/plots/" 
            year_dict_ = {'2024':[11, 12], '2025':[1,2]}
            generate_calendar_with_links_for_years(base_dir, year_dict_, 'L1')
        except Exception as e:
            pass

    except Exception as e:
        logging.error(f"An error occurred: {e}")   
        
else:
    pass
    #logging.info("No Observing segments")
