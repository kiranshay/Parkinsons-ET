from utils.constants import movement_dir
from utils.data_handling import load_all_files, get_data_from_txt_file
import pandas as pd

# Get file list
df = pd.concat(load_all_files(movement_dir))

# Filter for id = 001
df = df[df['subject_id'] == '001']
print(df.loc[0, :])

# Get file path of first record listed
file_name = df.loc[0, 'file_name']
file_path = movement_dir + file_name
print(file_path)
# ../movement/bins/001_Relaxed_LeftWrist.bin

# Get the channels for the selected record
channels = df.loc[0, 'channels']
print(channels)
# ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z']
n_channels = len(channels)

# Get the record data
record = get_data_from_txt_file(file_path, n_channels)
print(record.shape)
# (2048, 7)
