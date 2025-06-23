import warnings
from utils.constants import patient_dir
from utils.data_handling import load_all_files
import numpy as np
import pandas as pd

data_path = '../preprocessed/'

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

df.sort_values(by=['age'], inplace=True)

# The same number of samples from each class with each gender
ref_data1 = ['Female', 'Male',
             'Female', 'Male',
             'Female', 'Male']
ref_data2 = ['0', '0',
             '1', '1',
             '2', '2']
ref_data = np.array([ref_data1, ref_data2])

data = df.loc[:, ['gender', 'label']].values.transpose()


def _match_groups_encoding(data, ref_data):
    """
    Match group encoding: if e.g. one group has elements 'A' and 'C', and the other group 'A', 'B', 'C', the numerical
    encoding will be adjusted so that group B is counted with 0 values in the one group.
    """
    data_dict = set(data).union(set(ref_data))
    data_dict = list(data_dict)
    data_dict.sort()
    data = pd.Series(data).replace(data_dict, np.arange(len(data_dict))).values
    ref_data = pd.Series(ref_data).replace(data_dict, np.arange(len(data_dict))).values
    return data, ref_data


def _get_overlap_groups(data, ref_data):
    """
    Transform the elements of data to string. Concatenate the rows and join the string values to create new group
    labels.
    """
    data = np.array(data, dtype=str)
    ref_data = np.array(ref_data, dtype=str)
    data = list(map("_".join, zip(*data)))
    ref_data = list(map("_".join, zip(*ref_data)))
    data, ref_data = _match_groups_encoding(data, ref_data)
    return data, ref_data


def _get_groups_weights(data, ref_data):
    """
    Get the counts for each discrete group in the dataset and the resulting relative weights that should be applied to
    the samples in the original data to match the distribution in ref_data.
    """
    data = np.array(data, dtype=str)
    ref_data = np.array(ref_data, dtype=str)
    # data, ref_data = _match_groups_encoding(data, ref_data)
    data = np.array(data, dtype=int)
    if len(data.shape) > 1:
        raise ValueError(f'Wrong input: data must be 1d list or 1d ndarray. Passed: {data.shape}.')
    ref_data = np.array(ref_data, dtype=int)
    if len(ref_data.shape) > 1:
        raise ValueError(f'Wrong input: ref_data must be 1d list or 1d ndarray. Passed: {ref_data.shape}.')
    minlength = int(np.max([np.max(ref_data), np.max(data)]) + 1)
    counts = np.bincount(data, minlength=minlength)
    ref_counts = np.bincount(ref_data, minlength=minlength)
    weights = ref_counts / counts
    weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
    weights_per_sample = weights[data]
    return weights_per_sample, weights, counts, ref_counts


def _exact_stratified_sampling(data, weights, random_state=42):
    """
    Exactly match the objective distribution by randomly choosing the number of samples from each group that is coded
    as relative portions of the original length in weights.
    IMPORTANT NOTE: This method is only approximately giving the right result, as the values may become complex float
    numbers and slight numerical errors occur due to rounding etc.
    """
    num_to_sample = (weights / np.max(weights))
    num_to_sample = num_to_sample.astype(np.float32)
    rng = np.random.default_rng(random_state)
    idxs = []
    for n, frac in enumerate(num_to_sample):
        sub_idxs = np.argwhere(data == n)[:, 0]
        size = int(len(sub_idxs) * frac)
        random_idxs = rng.choice(sub_idxs, size=size, replace=False)
        idxs.extend(random_idxs)
    return idxs


def _iterative_stratified_sampling(data, ref_weights, frac=0.8, random_state=42):
    """
    Match the objective distribution by iteratively removing sample indices from the original data. The sampling is
    performed by calculating the group that has the highest difference when subtracting the relative proportion of the
    group in the original data set to the reference weights.
    """
    ref_weights = ref_weights / np.sum(ref_weights)
    rng = np.random.default_rng(random_state)
    orig_len = len(data)
    new_len = frac * orig_len
    idxs = np.arange(orig_len)
    while len(data) > new_len:
        weights = np.bincount(data, minlength=len(ref_weights))
        weights = weights / np.sum(weights)
        weights_diff = weights - ref_weights
        if not weights_diff.any():
            warnings.warn(f'Given frac {frac} is smaller than actual match {len(data) / orig_len}', UserWarning)
            break
        label_to_remove = np.argmax(weights_diff)
        idx_to_remove = np.where(data == label_to_remove)[0]
        idx_to_remove = rng.choice(idx_to_remove, size=1, replace=False)
        data = np.delete(data, idx_to_remove)
        idxs = np.delete(idxs, idx_to_remove)
    return idxs


data, ref_data = _get_overlap_groups(data, ref_data)

weights_per_sample, weights, counts, ref_counts = _get_groups_weights(data, ref_data)

# idxs = _exact_stratified_sampling(data, weights)

idxs = _iterative_stratified_sampling(data, ref_counts, frac=0.5)
idxs = idxs.tolist()

df = df.iloc[idxs, :]
df.to_csv(f'{data_path}stratified_subset_file_list.csv', index=False, sep=',')
