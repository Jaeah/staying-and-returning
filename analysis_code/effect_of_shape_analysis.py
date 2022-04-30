import pylab as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm import tqdm

import create_subjects_csvs
import stats_utils

# Number of TrackIt objects_
def _get_num_objects(dataset: str):
  """Returns the number of objects used in each trial of the given dataset."""
  if dataset == 'ORIGINAL':
    return 7
  if dataset == 'LABELING':
    return 5
  raise ValueError(f'dataset should be \'ORIGINAL\' or \'LABELING\' '
                   f'but was {dataset}')

def transition_type(HMM, object_idx):
  if object_idx == 0:
    return 'Transitions to Target'
  return 'Transitions to Distractors'

def run_analysis(dataset: str, coding: str):

  num_objects = _get_num_objects(dataset)
  
  # Load frame-level data
  df = create_subjects_csvs.get_frame_data(coding=coding, dataset=dataset)
  
  # Compute next object for each pair of consecutive frames.
  df = stats_utils.add_next_object_column(df)

  print('Computing frame-wise transition data over ~300K frames...')
  transitions_dict = {'is_transition': [], 'transition_type': [],
      'subject_id': [], 'source_shape': [], 'dest_shape': []}
  for idx, row in tqdm(df.iterrows()):
    for object_idx in range(num_objects):
      if object_idx != row['HMM']:

        transitions_dict['is_transition'].append(row['next_frame_HMM'] == object_idx)
        transitions_dict['transition_type'].append(transition_type(row['HMM'], object_idx))
        transitions_dict['subject_id'].append(row['subject_id'])
        transitions_dict['source_shape'].append(row['shapes'][row['HMM']])
        transitions_dict['dest_shape'].append(row['shapes'][object_idx])
  
  df = pd.DataFrame(transitions_dict)
  df = (df.groupby(['subject_id', 'dest_shape', 'transition_type'])
          .mean()
          .reset_index())

  sns.boxplot(x='dest_shape', y='is_transition', hue='transition_type', data=df)
  plt.legend(title='Transition Type')
  plt.xlabel('Destination Object Shape')
  plt.ylabel('Mean Transition Probability')

  model = ols('is_transition ~ C(dest_shape)*C(transition_type)', data=df).fit()
  anova_table = sm.stats.anova_lm(model)
  print(anova_table)

  plt.show()

if __name__ == '__main__':
  run_analysis('ORIGINAL', 'HMM')
