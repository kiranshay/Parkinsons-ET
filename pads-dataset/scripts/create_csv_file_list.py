import pandas as pd
from utils.constants import patient_dir
from utils.data_handling import load_all_files

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
df.to_csv(f'{data_path}file_list.csv', index=False, sep=',')
