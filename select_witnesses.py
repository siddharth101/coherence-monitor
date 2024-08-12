from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data,find_max_corr_channel,plot_max_corr_chan

from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lowfreq', type=float, help='lower boundary for frequncy in DeepClean')
parser.add_argument('--highfreq', type=float, help='upper boundary for frequncy in DeepClean')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory in which data is saved')
parser.add_argument('--ifo', default='L1', type=str, help='Interferometer')
args = parser.parse_args()

def give_group(a):
    group = a.split('_')[1]
    return group

time_ = int(args.savedir.split("/")[-2])
print('Current time is:', time_)

print('Start loading the *.csv files and doing the plot')
vals = plot_max_corr_chan(args.savedir, 10, args.ifo)

print('Print sorted list of witness channels to be copied to DeepClean')
vals_selc = vals.loc[(vals['frequency']>args.lowfreq) & (vals['frequency']<args.highfreq) & ((vals['corr1']>0.2) | (vals['corr2'] > 0.2))]

channels = []
channels1 = vals_selc.sort_values(['corr1'], ascending=False).drop_duplicates(['channel1'])['channel1'].to_list()
channels2 = vals_selc.sort_values(['corr2'], ascending=False).drop_duplicates(['channel2'])['channel2'].to_list()

channels = [c for c in channels1]
for c in channels2:
    if c not in channels1:
        channels.append(c)

with open('chanlist_o4.ini','w') as f:
    f.write(args.ifo+':GDS-CALIB_STRAIN\n')
    for c in channels:
        f.write(c[:-26].replace('_','-',1)+'\n')
