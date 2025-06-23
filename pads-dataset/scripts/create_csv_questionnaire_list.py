from utils.constants import questionnaire_dir
from utils.data_handling import load_all_files
import pandas as pd

data_path = '../preprocessed/'

# Store file list for ml project
df = pd.concat(load_all_files(questionnaire_dir))
df.to_csv(f'{data_path}questionnaire_file_list.csv', index=False, sep=',')
