import json
from glob import glob
import numpy as np
import pandas as pd
from .dict_handling import flatten_dict


def load_all_files(path, dataframe=True):
    """
    Load all .json files from the defined directory and return all the loaded meta data.

    Parameters
    ----------
    path : str
        Path to the directory holding the .json files.
    dataframe : bool, default = True
        Whether to flatten the meta data into dataframes.
    """
    data_list = []
    search_space = glob(path + '*json')
    search_space.sort()
    for f_name in search_space:
        with open(f_name, 'r') as f:
            data = json.load(f)
            if dataframe:
                data = flatten_dict(data)
                data = pd.DataFrame(data)
            data_list.append(data)
    return data_list


def get_data_from_txt_file(path, n_channels):
    record = np.loadtxt(path, dtype=np.float32, delimiter=",")
    return record


def get_data_from_questionnaire_response(meta_file):
    data = meta_file['answer'].values
    channels = (meta_file['questionnaire_name'] + '_' + meta_file['link_id']).values
    return data, channels


def get_data_from_observation(path, meta_file):
    all_records = []
    all_channels = []
    min_rows = meta_file['rows'].min()
    for idx, meta_item in meta_file.iterrows():
        n_splits = meta_item['rows'] // min_rows

        file_path = meta_item['file_name']
        record = get_data_from_txt_file(path + file_path, len(meta_item['channels']))
        record = np.swapaxes(record, 0, 1)
        channels = ['_'.join([meta_item['device_location'], channel]) for channel in meta_item['channels']]

        # Re-organize the raw data so that each record has the same length and all records fit into one matrix
        step = record.shape[1] // n_splits
        if n_splits > 1:
            new_record = []
            for n in range(0, record.shape[1], step):
                new_record.append(record[:, n:n+step])
            record = np.concatenate(new_record, axis=0)
            new_channels = []
            for n in range(n_splits):
                for channel in channels:
                    new_channels.append(f'{meta_item["record_name"]}{n+1}_{channel}')
            channels = new_channels
        else:
            channels = ['_'.join([meta_item['record_name'], channel]) for channel in channels]
        all_records.append(record)
        all_channels.extend(channels)

    all_records = np.concatenate(all_records, axis=0)

    return all_records, all_channels


def get_data(path):
    data_list = []
    channels_list = []
    meta_list = load_all_files(path, dataframe=True)
    for meta_file in meta_list:
        if meta_file['resource_type'].iloc[0] == 'questionnaire_response':
            data, channels = get_data_from_questionnaire_response(meta_file)
        elif meta_file['resource_type'].iloc[0] == 'observation':
            data, channels = get_data_from_observation(path, meta_file)
        else:
            raise Exception(f'The "resource_type" {meta_file["resource_type"].iloc[0]} could not be loaded.')
        data_list.append(data)
        channels_list.append(channels)
    return np.array(data_list, dtype=np.float32), channels_list
