import os
import glob
import pandas as pd
import unittest
from pandas.testing import assert_frame_equal
from utils import get_observing_segs, combine_data_files
from gwpy.segments import Segment, SegmentList

print("Imports done")

# Define segments and time range
segsl = SegmentList([Segment(1421395218.0, 1421400618.0)])
tstart = '2025-01-20 08:00:00'
tend = '2025-01-20 09:30:00'
# segl_ = get_observing_segs(t1=tstart, t2=tend, ifo='L1')

# Prepare folder path and load data
base_path = '/home/siddharth.soni/public_html/coherence_monitor'
folder_path = os.path.join(base_path, 'L1/2025-01-05/data/1420070718/')
files_ = glob.glob(folder_path + "*.csv")
nfiles = len(files_)

# Combine CSV files into a single DataFrame
li = []
frame = pd.DataFrame()
for i in range(nfiles):
    file = files_[i]
    channame = file.split(".csv")[-2].split("/")[-1]
    df = pd.read_csv(file, names=["freq", "value"])
    df.freq = df.freq.round(1)
    df.value = df.value.round(2)
    df["channel"] = channame
    if len(df) > 0:
        li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)

# Combine using utility function
fr = combine_data_files(folder_path)
print("Data prepared")


# Utility function for comparing DataFrames
def assert_dataframes_equal(test_case, df1, df2):
    try:
        assert_frame_equal(df1, df2)
    except AssertionError as e:
        test_case.fail(f"DataFrames are not equal: {e}")


# Unit test class
class TestUtils(unittest.TestCase):
    # Uncomment and implement this test if needed
    # def test_observing_segs(self):
    #     self.assertEqual(segsl, segl_)

    def test_combine_files(self):
        assert_dataframes_equal(self, frame, fr)


if __name__ == '__main__':
    unittest.main()
