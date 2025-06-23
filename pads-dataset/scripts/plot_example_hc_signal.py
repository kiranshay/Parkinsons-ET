import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import movement_dir
from utils.data_handling import load_all_files, get_data_from_txt_file

file_idx = f'{78:03d}'
channel_filter = 'Accelerometer'
file_filter = 'HoldWeight'

# Get file list
df = pd.concat(load_all_files(movement_dir))

# Filter for id = 034
df = df[df['subject_id'] == file_idx].reset_index()
df = df[df['file_name'].str.contains(file_filter)]
df = df.reset_index()

# Get file path of first record listed
file_name = df.loc[0, 'file_name']
file_path = movement_dir + file_name
channels = df.loc[0, 'channels']
n_channels = len(channels)
# Get the record data
x_l = get_data_from_txt_file(file_path, n_channels)
idxs = [idx for idx, channel in enumerate(channels) if channel_filter in channel]
x_l = x_l[:, idxs]

# Get file path of first record listed
file_name = df.loc[1, 'file_name']
file_path = movement_dir + file_name
channels = df.loc[1, 'channels']
n_channels = len(channels)
# Get the record data
x_r = get_data_from_txt_file(file_path, n_channels)
idxs = [idx for idx, channel in enumerate(channels) if channel_filter in channel]
x_r = x_r[:, idxs]
x = np.concatenate([x_l, x_r], axis=1)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True, sharey=True)

mov_data = x
sup_title = "HC subject during task 'Hold Weight'"

ax[0].plot(mov_data[:, :3], label=['x', 'y', 'z'])
ax[0].set_title('Left arm')
ax[0].set_ylabel('Acceleration [g]')
# ax[0].set_xlabel("Time in ms")
ax[0].set_xlim([0, 1024])
min_y = np.min(np.min(mov_data))
max_y = np.max(np.max(mov_data))
ax[0].set_ylim([-0.15, 0.15])
ax[0].legend(loc='upper right', ncol=3)


ax[1].plot(mov_data[:, 3:], label=['x', 'y', 'z'])
ax[1].set_title('Right arm')
ax[1].set_ylabel('Acceleration [g]')
ax[1].set_xlabel('Time [ms]')

plt.suptitle(sup_title)
plt.tight_layout()
plt.savefig(f'hc_{file_idx}_signal.jpg')
plt.close()