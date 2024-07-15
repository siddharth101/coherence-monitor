from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--date', type=str, help='YYYY-MM-DD')
parser.add_argument('--ifo', type=str, help='L1 or H1')
args = parser.parse_args()

t1 = args.date
ifo = args.ifo

date1 = datetime.strptime(t1, '%Y-%m-%d')
date2 = date1 + timedelta(days=1)
date2 = date2.strftime('%Y-%m-%d')


segs_ = get_observing_segs(date1, date2)
times_segs = get_times(seglist=segs_)

channel_path = 'channel_files/{}/'.format(ifo)

lsc_chans = pd.read_csv(channel_path + 'lsc_channels.csv', header=None, names=['channel'])
sus_chans = pd.read_csv(channel_path + 'sus_channels.csv', header=None, names=['channel'])
asc_chans = pd.read_csv(channel_path + 'asc_channels.csv', header=None, names=['channel'])
pem_chans = pd.read_csv(channel_path + 'pem_channels.csv', header=None, names=['channel'])
hepi_chans = pd.read_csv(channel_path + 'hepi_channels.csv', header=None, names=['channel'])
imc_chans = pd.read_csv(channel_path + 'imc_channels.csv', header=None, names=['channel'])
omc_chans = pd.read_csv(channel_path + 'omc_channels.csv', header=None, names=['channel'])
psl_sqz_tcs_chans = pd.read_csv(channel_path + 'psl_sqz_tcs_channels.csv', header=None, names=['channel'])
isi_chans = pd.read_csv(channel_path +  'isi_channels.csv', header=None, names=['channel'])

time_ = random.choice(times_segs)

print("Time is {}".format(time_))

ht_data = get_strain_data(time_, time_ + 900, ifo=ifo)
print("Got h(t) data")

def give_group(a):
    group = a.split('_')[1]
    
    return group

def get_coherence_asc(channel_list=asc_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    
    
    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return

def get_coherence_sus(channel_list=sus_chans['channel'], first_chan_index=0,
                     ifo=ifo, times=time_, strain_data=ht_data):
        
    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    channel_list = channel_list.iloc[first_chan_index:]
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return


def get_coherence_lsc(channel_list=lsc_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return 

def get_coherence_pem(channel_list=pem_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return 

def get_coherence_hepi(channel_list=hepi_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return

def get_coherence_psl_sqz_tcs(channel_list=psl_sqz_tcs_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return

def get_coherence_isi(channel_list=isi_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):


    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return

def get_coherence_imc(channel_list=imc_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return

def get_coherence_omc(channel_list=omc_chans['channel'], ifo=ifo, times=time_, strain_data=ht_data):

    files_ = get_frame_files(times, times + 900, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=times, 
                  endtime=times+900, ifo=ifo, strain_data=ht_data)
        
        
    return





p1 = multiprocessing.Process(target = get_coherence_asc)
p2 = multiprocessing.Process(target = get_coherence_lsc)
p3 = multiprocessing.Process(target = get_coherence_sus)
p4 = multiprocessing.Process(target = get_coherence_pem)
p5 = multiprocessing.Process(target = get_coherence_hepi)
p6 = multiprocessing.Process(target = get_coherence_psl_sqz_tcs)
p7 = multiprocessing.Process(target = get_coherence_isi)
p8 = multiprocessing.Process(target = get_coherence_imc)
p9 = multiprocessing.Process(target = get_coherence_omc)


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
p9.start()

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()
p9.join()

tac = time.time()
print(tac - tic)


file_path = '/home/siddharth.soni/O4/coherence_study/coherence_monitor/data/{}/'.format(int(time_))

vals = get_max_corr(file_path, save=True)
vals['group'] = vals['channel'].apply(give_group)

import plotly.express as px
import plotly
fig = px.scatter(vals, x="frequency", y="max_correlation", 
                  hover_data=['channel'], color= "group", labels={"max_correlation": "Max Correlation",
                                                                 "frequency":"Frequency [Hz]"})
plotly.offline.plot(fig, filename = 'plots/scatter_coh_{}.png'.format(int(time_)))
