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
parser.add_argument('--dur', type=float, default=1024.0, help='duration of data in secs')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory to save data')
args = parser.parse_args()

t1 = args.date
ifo = args.ifo
dur = args.dur
savedir = args.savedir

date1 = datetime.strptime(t1, '%Y-%m-%d')
date2 = date1 + timedelta(days=1)
date2 = date2.strftime('%Y-%m-%d')


segs_ = get_observing_segs(date1, date2)
times_segs = get_times(seglist=segs_)

channel_path = 'channel_files/{}/'.format(ifo)

df_all_chans = pd.read_csv(channel_path + '{}_all_chans.csv'.format(ifo), header=None, names=['channel'])

# df_chan1 = df_all_chans.iloc[:100]['channel']
# df_chan2 = df_all_chans.iloc[100:200]['channel']
# df_chan3 = df_all_chans.iloc[200:300]['channel']
# df_chan4 = df_all_chans.iloc[300:400]['channel']
# df_chan5 = df_all_chans.iloc[400:500]['channel']
# df_chan6 = df_all_chans.iloc[500:600]['channel']
# df_chan7 = df_all_chans.iloc[600:700]['channel']
# df_chan8 = df_all_chans.iloc[700:800]['channel']
# df_chan9 = df_all_chans.iloc[800:]['channel']

time_ = random.choice(times_segs)

print("Time is {}".format(time_))

ht_data = get_strain_data(time_, time_ + dur, ifo=ifo)

print("Got h(t) data between {} and {}".format(time_, time_+dur))

def give_group(a):
    group = a.split('_')[1]
    
    return group

def get_coherence_chan(channel_list, ifo=ifo, gpstime=time_, strain_data=ht_data, dur=dur):
   
    files_ = get_frame_files(gpstime, gpstime + dur, ifo=ifo)
    print("Got {} files".format(len(files_)))
    run_coherence(channel_list=channel_list, frame_files = files_, starttime=gpstime, 
                  endtime=gpstime+dur, ifo=ifo, strain_data=ht_data, savedir=savedir)
        
        
    return

p1 = multiprocessing.Process(target = get_coherence_chan,
                             args=(df_all_chans.iloc[:50]['channel'], ifo, time_, ht_data, dur))
p2 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[50:100]['channel'], ifo, time_, ht_data, dur))
p3 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[100:150]['channel'], ifo, time_, ht_data, dur))
p4 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[150:200]['channel'], ifo, time_, ht_data, dur))
p5 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[200:250]['channel'], ifo, time_, ht_data, dur))
p6 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[250:300]['channel'], ifo, time_, ht_data, dur))
p7 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[300:350]['channel'], ifo, time_, ht_data, dur))
p8 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[350:400]['channel'], ifo, time_, ht_data, dur))
p9 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[400:450]['channel'], ifo, time_, ht_data, dur))
p10 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[450:500]['channel'], ifo, time_, ht_data, dur))
p11 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[500:550]['channel'], ifo, time_, ht_data, dur))
p12 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[550:600]['channel'], ifo, time_, ht_data, dur))
p13 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[600:650]['channel'], ifo, time_, ht_data, dur))
p14 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[650:700]['channel'], ifo, time_, ht_data, dur))
p15 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[700:750]['channel'], ifo, time_, ht_data, dur))
p16 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[750:800]['channel'], ifo, time_, ht_data, dur))
p17 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[800:850]['channel'], ifo, time_, ht_data, dur))
p18 = multiprocessing.Process(target = get_coherence_chan, 
                             args=(df_all_chans.iloc[850:950]['channel'], ifo, time_, ht_data, dur))






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
p10.start()
p11.start()
p12.start()
p13.start()
p14.start()
p15.start()
p16.start()
p17.start()
p18.start()

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()
p9.join()
p10.join()
p11.join()
p12.join()
p13.join()
p14.join()
p15.join()
p16.join()
p17.join()
p18.join()


tac = time.time()
print(tac - tic)


#file_path = '/home/siddharth.soni/O4/coherence_study/coherence_monitor/data/{}/'.format(int(time_))
#file_path = os.path.join(savedir, '{}'.format(int(time_)), '')

#vals = get_max_corr(file_path, save=True)
#vals['group'] = vals['channel'].apply(give_group)
#
#import plotly.express as px
#import plotly
#fig = px.scatter(vals, x="frequency", y="max_correlation", 
#                  hover_data=['channel'], color= "group", labels={"max_correlation": "Max Correlation",
#                                                                 "frequency":"Frequency [Hz]"})
#fig.update_layout(
#    title=dict(text="Max Coherence during {} -- {}".format(str(time_), str(time_ + 900)), font=dict(
#        family="Courier New, monospace",
#        size=18,
#        color="RebeccaPurple")))
#
#
#plotly.offline.plot(fig, filename = 'plots/scatter_coh_{}.png'.format(int(time_)))
