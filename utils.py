"""Utilities for coherence monitor"""

from gwpy.time import to_gps, from_gps
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import gwdatafind
import numpy as np
import pandas as pd
import os
import glob
import plotly.express as px
import plotly
import multiprocessing
import concurrent.futures

__author__ = "Siddharth Soni <siddharth.soni@ligo.org>"


def give_group_v2(a):
    group = a.split(":")[1].split("-")[0]
    return group


def get_strain_data(starttime, endtime, ifo="L1"):
    ht = TimeSeries.get(f"{ifo}:GDS-CALIB_STRAIN", starttime, endtime)
    return ht


def get_frame_files(starttime, endtime, ifo, host=None):
    site_ = ifo[0]
    if host:
        files = gwdatafind.find_urls(f"{site_}", f"{ifo}_R",
                                     starttime, endtime, host)
    else:
        files = gwdatafind.find_urls(f"{site_}", f"{ifo}_R",
                                     starttime, endtime)

    return sorted(files)


def get_unsafe_channels(ifo):
    path_home = "/home/siddharth.soni/src/coherence-monitor/channel_files/"
    path = os.path.join(path_home, f"{ifo}/{ifo}_unsafe_channels.csv")
    return pd.read_csv(path)


def get_observing_segs(t1, t2, ifo):
    segs = DataQualityFlag.query(f"{ifo}:DMT-ANALYSIS_READY:1", t1, t2)
    seg_list = SegmentList()

    for seg in segs.active:
        if seg.end - seg.start > 3600:
            seg_list.append(seg)

    return seg_list


def get_times(seglist, duration=3600):
    times = [np.arange(i.start, i.end - duration, duration) for i in seglist]
    return [item for sublist in times for item in sublist]


def calc_coherence(
    channel2,
    start_time,
    end_time,
    fft,
    overlap,
    strain_data,
    frame_file=None,
    channel1=None,
):
    t1 = to_gps(start_time)
    t2 = to_gps(end_time)
    if frame_file:
        ts2 = TimeSeries.read(frame_file, channel=channel2, start=t1, end=t2)
    else:
        ts2 = TimeSeries.fetch(channel2, start=t1, end=t2)

    if channel1:
        ts1 = TimeSeries.fetch(channel1, t1, t2)
    else:
        ts1 = strain_data

    ts1 = ts1.resample(ts2.sample_rate)
    coh = ts1.coherence(ts2, fftlength=fft, overlap=overlap)

    for i in np.where(coh.value == np.inf)[0]:
        try:
            coh.value[i] = (coh.value[i - 2] + coh.value[i - 1]) / 2
        except IndexError:
            coh.value[i] = 1e-20

    return coh


def run_coherence(
    channel_list,
    frame_files,
    starttime,
    endtime,
    strain_data,
    savedir,
    coh_thresh,
    ifo="L1",
    timeout=30,
):
    t1, t2 = to_gps(starttime), to_gps(endtime)
    savedir = os.path.join(savedir, f"{t1}", "")

    if not os.path.exists(savedir):
        print(f"Creating the output dir {savedir}")
        os.makedirs(savedir)

    for channel in channel_list:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                calc_coherence,
                strain_data=strain_data,
                channel1=None,
                channel2=channel,
                frame_file=frame_files,
                start_time=t1,
                end_time=t2,
                fft=10,
                overlap=5,
            )

            try:
                coh = future.result(timeout=timeout)
                coh_ = coh[coh.value > coh_thresh]
                if len(coh_) > 0:
                    coh_.write(f"{savedir}{channel}.csv")

            except concurrent.futures.TimeoutError:
                print(f"Calculation for channel {channel} timed out.")
                continue


def get_max_corr(output_dir, save=False):
    files = glob.glob(f"{output_dir}*.csv")
    vals = []

    for file in files:
        chan_name = file.split("/")[-1].split("DQ")[0] + "DQ"
        fs = FrequencySeries.read(file)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(1 / n_diff), int(200 / n_diff)
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value

        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals = pd.DataFrame(
                vals, columns=["channel", "max_correlation", "frequency"]
            )
            df_vals_ = df_vals[df_vals["max_correlation"] > 0]
        else:
            vals.append(-1)

    return df_vals_


def get_max_corr_band(output_dir, flow=10, fhigh=20, save=False):
    files = glob.glob(f"{output_dir}*.csv")
    vals = []

    for file in files:
        chan_name = file.split("/")[-1].split("DQ")[0] + "DQ"
        fs = FrequencySeries.read(file)
        n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
        n_diff = n2 - n1
        ind1, ind2 = int(flow / n_diff), int(fhigh / n_diff)
        fs_ = fs[ind1:ind2]
        max_value = fs_.max().value
        max_value_frequency = fs_.frequencies[fs_.argmax()].value

        if save:
            vals.append((chan_name, max_value, max_value_frequency))
            df_vals = pd.DataFrame(
                vals, columns=["channel", "max_correlation", "frequency"]
            )
            df_vals_ = df_vals[df_vals["max_correlation"] > 0]
        else:
            vals.append(-1)

    return df_vals_


def combine_csv(dir_path, ifo):
    all_files = glob.glob(f"{dir_path}*.csv")
    chan_removes = get_unsafe_channels(ifo=ifo)["channel"]

    for j in chan_removes:
        all_file = [i for i in all_files if not i.startswith(f"{dir_path}{j}")]

    li = []
    fns = []

    for filename in all_file:
        fn = filename.split("/")[-1].split("_14")[0] + "_freq"
        fn_val = filename.split("/")[-1].split("_14")[0] + "_corr"
        fns.append(fn)
        fns.append(fn_val)
        df = pd.read_csv(filename, index_col=None, header=None)
        li.append(df)

    frame = pd.concat(li, axis=1, ignore_index=True)
    frame.columns = fns
    return frame


def check_channel_coherence(channel, ifo, t1, t2, fft=10, overlap=5):

    files = get_frame_files(starttime=t1, endtime=t2, ifo=ifo)
    ts_aux = TimeSeries.read(files, channel=channel, start=t1, end=t2)
    ts_gds = TimeSeries.fetch("{}:GDS-CALIB_STRAIN".format(ifo), t1, t2)

    ts_gds = ts_gds.resample(ts_aux.sample_rate)
    coh = ts_gds.coherence(ts_aux, fftlength=fft, overlap=overlap)

    return coh


def create_dataframe(files, threshold=0.6, frequency_threshold=200):

    coh_threshold = threshold
    freq_threshold = frequency_threshold

    df_list = []
    for file in files:
        df = pd.read_csv(file, header=None, names=["frequency", "coherence"])
        df = df[(df["coherence"] > coh_threshold) &
                (df["frequency"] < freq_threshold)]
        if len(df) != 0:
            chan_name = file.split("/")[-1].split(".csv")[0]
            df["channel"] = [chan_name] * len(df)
            df_list.append(df)

    if df_list:
        frame = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        frame = pd.DataFrame()

    return frame


def freq_masks(df):
    freq_mask = (
        ((df.frequency >= 59.8) & (df.frequency <= 60.2))
        | ((df.frequency >= 119.8) & (df.frequency <= 120.2))
        | ((df.frequency >= 179.8) & (df.frequency <= 180.2))
    )

    return df[~freq_mask]


def coherence_above(ifo, date, path=None):

    gpstime = to_gps(date)
    # Define the path if not provided
    basepath = "home/siddharth.soni/public_html/coherence_monitor/"
    if path is None:
        path = os.path.join(basepath, f"{ifo}/{date}/{gpstime}/data/")
    times = [os.path.join(path, i, "") for i in os.listdir(path)]

    if not times:
        print("No data found for this date")
    for time in times:
        gpstime_ = time.split("/")[-2]
        # print(gpstime_)
        print(f"Running for {time}")
        files = glob.glob(time + "*.csv")
        df = create_dataframe(files)
        df = freq_masks(df)
        df["time"] = gpstime_
        df.reset_index(drop=True, inplace=True)

        # print(len(df))
        if len(df) != 0:
            maxpath = path.replace("data/", "")
            path_ = os.path.join(maxpath, "max_coherence", "")
            os.makedirs(path_, exist_ok=True)
            # print(path_)
            df.to_csv(path_ + f"coherence_above_{gpstime_}.csv", index=None)

    return


def combine_data_files(path):

    files = glob.glob(path + "*.csv")

    nfiles = len(files)
    if nfiles == 0:  # If no files are found
        return pd.DataFrame()  # Return an empty DataFrame
    li = []
    frame = pd.DataFrame()
    for i in range(nfiles):
        file = files[i]
        channame = file.split(".csv")[-2].split("/")[-1]
        df = pd.read_csv(file, names=["freq", "value"])
        df.freq = df.freq.round(1)
        df.value = df.value.round(2)
        df["channel"] = channame
        if len(df) > 0:
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame


def get_max_coherence(dataframe, min_freq=0.0, max_freq=100.0, fft=10):

    res = 1 / fft
    frequencies = np.arange(min_freq, max_freq + res, res)
    df = dataframe

    limaxdf = []
    for frequency in frequencies:
        frequency = round(frequency, 1)
        try:
            dfmax = (
                df[df["freq"] == frequency]
                .sort_values(by="value", ascending=False)
                .iloc[0]
            )
            limaxdf.append(dfmax)
        except Exception:
            pass

    frame_limaxdf = pd.concat(limaxdf, axis=1, ignore_index=True).T

    return frame_limaxdf


def generate_plots(date, ifo):
    """
    Generate coherence plots for a given date and IFO.

    Parameters:
    - date: str, the date for which plots are generated
    - ifo: str, interferometer ("L1" or "H1")

    Returns:
    - None
    """
    # Define the base path for the IFO
    base_path = "/home/siddharth.soni/public_html/coherence_monitor/"
    pathifo = os.path.join(base_path, ifo)

    # Define the folder path containing data
    folder_path = os.path.join(pathifo, date, "data")
    gps_folders = os.listdir(folder_path)

    for folder in gps_folders:
        gps_folder_path = os.path.join(folder_path, folder, '')
        print(gps_folder_path)

        # Process data files and compute max coherence
        try:
            fr = combine_data_files(gps_folder_path)
            frmax = get_max_coherence(fr, min_freq=0.0, max_freq=200.0, fft=10)
            vals = frmax
            vals["group"] = vals["channel"].apply(give_group_v2)
            vals.rename(
                columns={"value": "Coherence", "freq": "Frequency"},
                inplace=True,
            )

            # Define the plot directory
            plotdir = os.path.join(pathifo, date, "plots", folder)
            os.makedirs(plotdir, exist_ok=True)
            print(plotdir)

            # Create the scatter plot
            fig1 = px.scatter(
                vals,
                x="Frequency",
                y="Coherence",
                hover_data=["channel"],
                color="group",
                labels={
                    "max_correlation": "Max Coherence",
                    "frequency": "Frequency [Hz]",
                },
            )

            # Update the plot layout
            fig1.update_layout(
                title=dict(
                    text=(
                        f"{ifo}: Highest Coherence channel at each frequency "
                        f"during {folder} -- {str(int(folder) + 1024)}"
                    ),
                    font=dict(
                        family="Courier New, monospace",
                        size=22,
                        color="RebeccaPurple",
                    ),
                )
            )

            # Save the plot
            plotly.offline.plot(
                fig1,
                filename=os.path.join(plotdir, f"channels_coh_{int(folder)}.png"),
            )
        except Exception as e:
            pass

    return


def make_plots(folder, output, ifo):

    folder_t = folder.split("/")[-2]
    plotdir = output
    fr = combine_data_files(folder)
    frmax = get_max_coherence(fr, min_freq=0.0, max_freq=200.0, fft=10)
    vals = frmax
    vals["group"] = vals["channel"].apply(give_group_v2)
    vals.rename(columns={"value": "Coherence", "freq": "Frequency"},
                inplace=True)
    os.makedirs(plotdir, exist_ok=True)
    print(plotdir)
    fig1 = px.scatter(
        vals,
        x="Frequency",
        y="Coherence",
        hover_data=["channel"],
        color="group",
        labels={"max_correlation": "Max Coherence", "frequency":
                "Frequency [Hz]"},
    )

    fig1.update_layout(
       title=dict(
                text=(
                    f"{ifo}: Highest Coherence channel at each frequency "
                    f"during {folder} -- {str(int(folder) + 1024)}"
                ),
                font=dict(
                    family="Courier New, monospace",
                    size=22,
                    color="RebeccaPurple",
                ),
            )
        )
    plotly.offline.plot(
                        fig1,
                        filename=f"{plotdir}channels_coh_{int(folder_t)}.png"
                        )

    return


# Utilities to read data for any given date


def get_day_files(date, ifo):

    if ifo == "H1":
        pathifo = "/home/siddharth.soni/public_html/coherence_monitor/H1/"
    else:
        pathifo = "/home/siddharth.soni/public_html/coherence_monitor/L1/"

    folder_path = os.path.join(pathifo, date, "data", "")
    gps_folders = os.listdir(folder_path)

    files_folders = {}
    for folder in gps_folders:
        files = glob.glob(os.path.join(folder_path, folder, "") + "*.csv")
        files_folders[folder] = files

    return files_folders


def get_day_data(date, ifo, mask=True):

    day_files = get_day_files(date, ifo)

    li = []
    print(f"Getting coherence data for {ifo} on {date}")
    for key in list(day_files.keys()):
        print(key)
        files = day_files[key]
        for file in files:
            channame = file.split(".csv")[-2].split("/")[-1]
            df = pd.read_csv(file, names=["frequency", "coherence"])
            df["gpstime"] = int(key)
            df["date"] = date
            df["channel"] = channame
            li.append(df)

    if len(li) > 0:
        frame_ = pd.concat(li, axis=0, ignore_index=True)
    else:
        frame_ = pd.DataFrame()

    if mask and len(li) > 0:
        frame_ = freq_masks(frame_)
        frame_.reset_index(drop=True, inplace=True)
    else:
        pass

    return frame_


# Utilities to read day data using multiprocessing


def read_data_folder(day_files, n1=0, n2=2, mask=True):

    folders = list(day_files.keys())[n1:n2]
    li = []
    for folder in folders:
        files = day_files[folder]
        date = from_gps(folder).strftime("%Y-%m-%d")
        print(f"Reading data for {folder}")
        for file in files:

            channame = file.split(".csv")[-2].split("/")[-1]
            df = pd.read_csv(file, names=["frequency", "coherence"])
            df["gpstime"] = int(folder)
            df["date"] = date
            df["channel"] = channame
            li.append(df)

        if len(li) > 0:
            frame_ = pd.concat(li, axis=0, ignore_index=True)
        else:
            frame_ = pd.DataFrame()

        if mask and len(li) > 0:
            frame_ = freq_masks(frame_)
            frame_.reset_index(drop=True, inplace=True)
        else:
            pass

    return frame_


def run_process_day_data(date, ifo):
    day_files_ = get_day_files(date, ifo)

    n = len(day_files_.keys())
    # Manager list to collect DataFrame results
    with multiprocessing.Manager() as manager:
        results = manager.list()

        def worker_wrapper(*args):
            # Call read_data_folder and append its result to the shared list
            result = read_data_folder(*args)
            results.append(result)

        # Create processes
        processes = [
            multiprocessing.Process(
                target=worker_wrapper, args=(day_files_, i, i + 2, True)
            )
            for i in range(0, n, 2)
        ]

        # Start all processes
        [p.start() for p in processes]

        # Wait for all processes to finish
        [p.join() for p in processes]

        # Convert results (manager.list) back to a Python list
        frame_concat = []
        if len(list(results)) > 0:
            frame_concat = pd.concat(list(results), axis=0, ignore_index=True)
        return frame_concat


### Utilities to calculate the daily average and plot them
def daily_average(date, ifo,):

    if ifo == "H1":
        pathifo = "/home/siddharth.soni/public_html/coherence_monitor/H1/"
    else:
        pathifo = "/home/siddharth.soni/public_html/coherence_monitor/L1/"

    folder_path = os.path.join(pathifo, date, "")

    frame = run_process_day_data(ifo=ifo, date=date)
    try:
        frame = frame[frame.frequency<=200]

        frequencies = frame.frequency.unique()


        vals = []
        for i in frequencies:
            mean = round(frame[frame.frequency==i]['coherence'].mean(),2)
            most_freq_channel = frame[frame.frequency==i]['channel'].value_counts().index[0]
            vals.append((round(i,1),mean, most_freq_channel, date))

        if len(vals)>0:
            df = pd.DataFrame(vals, columns=['frequency', 'mean_coh', 'channel', 'date'])
            df.sort_values(by='frequency', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(folder_path+'mean_coherence.csv', index=None)
            return df
    except Exception as e:
        pass

    return


def make_plot_meancoh(date, ifo):

    if ifo == "H1":
        pathifo = "/home/siddharth.soni/public_html/coherence_monitor/H1/"
    else:
        pathifo = "/home/siddharth.soni/public_html/coherence_monitor/L1/"

    folder_path = os.path.join(pathifo, date, "")
    vals = pd.read_csv(folder_path + 'mean_coherence.csv')
    vals["group"] = vals["channel"].apply(give_group_v2)

    fig1 = px.scatter(
        vals,
        x="frequency",
        y="mean_coh",
        hover_data=["channel"],
        color="group",
        labels={"mean_coh": "Mean Coherence", "frequency":
                "Frequency [Hz]"},
    )

    fig1.update_layout(
       title=dict(
                text=(
                    f"{ifo}: Mean Coherence at each frequency "
                    f" on {date}"
                ),
                font=dict(
                    family="Courier New, monospace",
                    size=22,
                    color="RebeccaPurple",
                ),
        ),
        yaxis_range = [0,1])
    plotly.offline.plot(
                        fig1,
                        filename=f"{folder_path}mean_coh.png"
                        )

    return





import os
import calendar
from datetime import datetime
from gwpy.time import to_gps, from_gps

def generate_calendar_with_links_for_years(base_dir, year_dict, ifo):
    base_dir = os.path.join(base_dir, ifo, '')
    title = 'CohMon Plots'

    if ifo == 'L1':
        date_color = 'dodgerblue'
        link_color = 'cornflowerblue'
    else:
        date_color = 'tomato'
        link_color = 'salmon'

    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calendar</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            .calendar {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }}
            .month {{
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                text-align: center;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            th {{
                background-color: #f4f4f4;
            }}
            td {{
                height: 50px;
                vertical-align: top;
                position: relative;
            }}
            .day {{
                font-weight: bold;
            }}
            .link {{
                text-decoration: none;
                color: {date_color};
                cursor: pointer;
            }}
            .link:hover {{
                text-decoration: underline;
            }}
            .time-links {{
                display: none;
                position: absolute;
                top: 50px;
                left: 0;
                background-color: white;
                border: 1px solid #ddd;
                padding: 10px;
                z-index: 10;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .time-links a {{
                display: block;
                color: {link_color};
                margin-bottom: 5px;
            }}
            .time-links a:hover {{
                text-decoration: underline;
            }}
        </style>
        <script>
            function toggleVisibility(dateId) {{
                const timeLinks = document.querySelectorAll('.time-links');
                timeLinks.forEach(link => {{
                    if (link.id !== dateId) {{
                        link.style.display = "none";
                    }}
                }});
                const element = document.getElementById(dateId);
                if (element.style.display === "none" || element.style.display === "") {{
                    element.style.display = "block";
                }} else {{
                    element.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <h1>{ifo} {title}</h1>
        <div class="calendar">
    """
    #year_dict = {'2024':[11, 12], '2025':[1]}
    # Generate calendar content for each year
    for year_ in year_dict.keys():
        year = int(year_)
        html_content += f"<h2>Year: {year}</h2>"
        existing_folders = set(os.listdir(base_dir))
        
        for month in year_dict[str(year)]:
            #year = int(year_)
            print(year, month)
            print(type(year), type(month))
            month_name = calendar.month_name[month]
            _, num_days = calendar.monthrange(year, month)

            # Start a month container
            html_content += f"""
            <div class="month">
                <h2>{month_name}</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Sun</th>
                            <th>Mon</th>
                            <th>Tue</th>
                            <th>Wed</th>
                            <th>Thu</th>
                            <th>Fri</th>
                            <th>Sat</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            # Generate days in calendar format
            cal = calendar.Calendar()
            month_days = cal.monthdayscalendar(year, month)
            for week in month_days:
                html_content += "<tr>"
                for day in week:
                    if day == 0:
                        html_content += "<td></td>"
                    else:
                        folder_name = f"{year}-{month:02d}-{day:02d}"
                        if folder_name in existing_folders:
                            time_folders_html = ""
                            time_folder_path = os.path.join(base_dir, folder_name)
                            if os.path.isdir(time_folder_path):
                                for time_folder in sorted(os.listdir(time_folder_path)):
                                    time_folder_full_path = os.path.join(
                                        time_folder_path, time_folder
                                    )
                                    if os.path.isdir(time_folder_full_path) and time_folder.isdigit():
                                        timestamp = int(time_folder)
                                        time_str = from_gps(timestamp).strftime("%H:%M")
                                        files = os.listdir(time_folder_full_path)
                                        if files:
                                            file_full_path = files[0]
                                            time_folders_html += f'<a href="{folder_name}/{time_folder}/{file_full_path}" class="link">{time_str}</a>'

                            html_content += f"""
                                <td>
                                    <div>
                                        <a class="link" onclick="toggleVisibility('{folder_name}')">{day}</a>
                                    </div>
                                    <div id="{folder_name}" class="time-links">
                                        {time_folders_html}
                                    </div>
                                </td>
                            """
                        else:
                            html_content += f"<td>{day}</td>"
                html_content += "</tr>"

            # Close the month container
            html_content += """
                    </tbody>
                </table>
            </div>
            """

        # End HTML content
        html_content += """
            </div>
        </body>
        </html>
        """

    # Save to file
    output_path = os.path.join(base_dir, "calendar_combined.html")
    with open(output_path, "w") as file:
        file.write(html_content)

    print(f"Combined calendar HTML file created: {output_path}")
    
    return

