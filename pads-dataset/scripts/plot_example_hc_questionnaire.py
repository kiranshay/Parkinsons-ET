import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

preprocessed_dir = '../preprocessed/'
file_idx = 78

# Channels
channels = ['Dribbiling', 'Taste/smelling', 'Swallowing', 'Vomiting', 'Constipation',
            'Bowel inconsistence', 'Bowel emptying incomplete', 'Urgency', 'Nocturia', 'Pains',
            'Weight', 'Remembering', 'Loss of interest', 'Hallucinations', 'Concentrating',
            'Sad, blues', 'Anxiety', 'Sex drive', 'Sex difficulty', 'Dizzy',
            'Falling', 'Daytime sleepiness', 'Insomnia', 'Intense vivid dreams', 'Acting out during dreams',
            'Restless legs', 'Swelling', 'Sweating', 'Diplopia', 'Delusions']

# Get file list
df = pd.read_csv(f'{preprocessed_dir}file_list.csv')

# Filter for id = 034
subject = df[df['id'] == file_idx].reset_index().loc[0, :]
print(subject)

# Get data
data = np.fromfile(f'{preprocessed_dir}/questionnaire/{file_idx:03d}_ml.bin', dtype=np.float32)
data = np.where(data == 1.0, 'yes', 'no')
colors = np.where(data == 'yes', 'blue', 'red')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3))

ax.set_title('HC subject questionnaire')
ax.scatter(range(len(data)), data, marker='s', c=colors)

plt.xticks(range(len(channels)), channels, rotation=90)

plt.tight_layout()
plt.savefig(f'hc_{file_idx:03d}_questionnaire.jpg')
plt.close()
