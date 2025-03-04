import pandas as pd

filename = "L1-O4-standard.ini"

ifo = 'L1'
with open(filename, "r") as file:
    lines = file.readlines()
    
safe_lines = [line.strip() for line in lines if f'{ifo}:' in line and ' safe' in line and ' glitchy' not in line] 
unsafe_lines = [line.strip() for line in lines if f'{ifo}:' in line and 'unsafe' in line]

data_safe = []
for line in safe_lines:
    row = line.split()
    data_safe.append(row)
    
data_unsafe = []
for line in unsafe_lines:
    row = line.split()
    data_unsafe.append(row)
    
    
df_safe = pd.DataFrame(data_safe, columns=['channel', 'sample_rate', 'safety', 'cleanliness'])
df_unsafe = pd.DataFrame(data_unsafe, columns=['channel', 'sample_rate', 'safety', 'cleanliness'])

df_safe['channel'].to_csv(f'{ifo}_safe_channels.csv', index=None)
df_unsafe['channel'].to_csv(f'{ifo}_unsafe_channels.csv', index=None)
