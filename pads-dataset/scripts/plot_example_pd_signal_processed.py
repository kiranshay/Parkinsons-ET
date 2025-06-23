import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

preprocessed_dir = '../preprocessed/'
file_idx = 60
channel_filter = 'HoldWeight_Acceleration'

# Channels
channels = []
for task in ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
             "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]:
    for device_location in ["LeftWrist", "RightWrist"]:
        for sensor in ["Acceleration", "Rotation"]:
            for axis in ["X", "Y", "Z"]:
                channel = f"{task}_{sensor}_{device_location}_{axis}"
                channels.append(channel)

# Get file list
df = pd.read_csv(f'{preprocessed_dir}file_list.csv')

# Filter for id = 034
subject = df[df['id'] == file_idx].reset_index().loc[0, :]
print(subject)

# Get data
x = np.fromfile(f"{preprocessed_dir}/movement/{file_idx:03d}_ml.bin", dtype=np.float32).reshape((-1, 976))
# Filter for specific task
idxs = [idx for idx, channel in enumerate(channels) if channel_filter in channel]
x = x[idxs, :]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True, sharey=True)

mov_data = np.swapaxes(x, 0, 1)
sup_title = "PD subject during task 'Hold Weight'"

ax[0].plot(mov_data[:, :3], label=['x', 'y', 'z'])
ax[0].set_title('Left arm')
ax[0].set_ylabel('Acceleration [g]')
# ax[0].set_xlabel("Time in ms")
ax[0].set_xlim([0, 976])
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
plt.savefig(f'pd_{file_idx:03d}_signal_processed.jpg')
plt.close()
