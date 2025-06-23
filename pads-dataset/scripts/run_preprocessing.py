import multiprocessing
import os
from pathlib import Path
import pandas as pd
from utils.l1_trend_filter import l1_trend_filter
from utils.constants import movement_dir, questionnaire_dir, patient_dir
from utils.data_handling import load_all_files, get_data, get_data_from_observation
import numpy as np

data_path = '../preprocessed/'
quest_path = data_path + '/questionnaire/'
Path(quest_path).mkdir(parents=True, exist_ok=True)
mov_path = data_path + '/movement/'
Path(mov_path).mkdir(parents=True, exist_ok=True)


def preprocess_movement(df, overwrite=False):
    id = df['subject_id'][0]
    data, channels = get_data_from_observation(movement_dir, df)

    channels_sorted = []
    # Sort by the following pattern
    for task in ['Relaxed1', 'Relaxed2', 'RelaxedTask1', 'RelaxedTask2', 'StretchHold', 'LiftHold', 'HoldWeight',
                 'PointFinger', 'DrinkGlas', 'CrossArms', 'TouchIndex', 'TouchNose', 'Entrainment1', 'Entrainment2']:
        for wrist in ['LeftWrist', 'RightWrist']:
            for sensor in ['Time', 'Accelerometer', 'Gyroscope']:
                if sensor == 'Time':
                    channel_name = '_'.join([task, wrist, sensor])
                    channels_sorted.append(channel_name)
                else:
                    for axis in ['X', 'Y', 'Z']:
                        channel_name = '_'.join([task, wrist, sensor, axis])
                        channels_sorted.append(channel_name)

    sorting_indices = [channels.index(channel_name) for channel_name in channels_sorted]

    data = data[sorting_indices]
    channels = np.array(channels)[sorting_indices]

    to_remove = 'Time|LiftHold|PointFinger|TouchIndex'
    keep_mask = ~pd.Series(channels).str.contains(to_remove)
    channels = channels[keep_mask]

    to_process = 'Accelerometer'
    process_mask = pd.Series(channels).str.contains(to_process)

    # Check if file already exists
    if not overwrite:
        all_files = os.listdir(mov_path)
        all_files = list(filter(lambda f: f.endswith('.bin'), all_files))
        if f'{id}_ml.bin' in all_files:
            return

    # Remove assessment steps
    data = data[keep_mask]
    # Remove gravitational offset
    data[process_mask, :] = np.apply_along_axis(lambda x: x - l1_trend_filter(x, vlambda=50, verbose=False), 1,
                                                data[process_mask, :])
    # Remove first half second of the signal (vibration notification)
    data = data[:, 48:]
    data.tofile(f'{mov_path}{id}_ml.bin')


if __name__ == '__main__':
    # Store file list for ml project
    df = pd.concat(load_all_files(patient_dir))
    df['label'] = df['condition']
    df.replace({'label': {'Healthy': 0,
                          "Parkinson's": 1,
                          'Other Movement Disorders': 2,
                          'Essential Tremor': 2,
                          'Multiple Sclerosis': 2,
                          'Atypical Parkinsonism': 2}},
               inplace=True)
    df.to_csv(f'{data_path}file_list.csv', index=False, sep=',')

    # Store all questionnaire data for ml project
    data, channels = get_data(questionnaire_dir)
    for idx, data_sample in enumerate(data):
        data_sample.tofile(f'{quest_path}{idx + 1:03d}_ml.bin')

    # Store file list for ml project
    df_list = load_all_files(movement_dir)
    # Run in parallel
    for df_element in df_list:
        preprocess_movement(df_element)
