import math

import pylab as plt
import numpy as np
import pandas as pd
import random
from skmisc.loess import loess
import seaborn as sns
import statsmodels.api as sm

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

def _get_max_distance(dataset: str):
  """Returns the maximum usable distance value in the given dataset."""
  if dataset == 'ORIGINAL':
    return 1000
  if dataset == 'LABELING':
    return 700
  raise ValueError(f'dataset should be \'ORIGINAL\' or \'LABELING\' '
                   f'but was {dataset}')

def _get_corrs_by_subject(df: pd.DataFrame, response: str):
  """Correlations (by subject) between inter-object distance and response."""
  return df.groupby('subject_id')[['distance_in_degrees', response]].corr().iloc[0::2, -1]

def _get_corrs_by_age(df: pd.DataFrame, response: str):
  """Correlations (by subject) between inter-object distance and response."""
  return df.groupby(['subject_id', 'age'])[['distance_in_degrees', response]].corr().iloc[0::2, -1]

# Plot transition likelihood as a function of inter-object distance, separately
# for each transition type
def _plot_loess(x, y, plt_idx, dataset, subsample_proportion):

  # Sort data by x-coordinate for plotting
  ind = np.argsort(x)
  x = x[ind]
  y = y[ind]

  l = loess(x, y, surface='direct')
  l.fit()
  pred = l.predict(x, stderror=True)
  conf = pred.confidence(alpha=0.01)
  
  lowess = pred.values
  ll = np.maximum(0, conf.lower)
  ul = np.minimum(1, conf.upper)
  
  plt.subplot(3, 1, plt_idx)
  plt.plot(x, y, '+')
  plt.plot(x, lowess)
  plt.xlim(left=0, right=stats_utils.pixels_to_degrees(_get_max_distance(dataset)))
  y_margin = subsample_proportion/20
  plt.ylim(bottom=-y_margin, top=subsample_proportion+y_margin)
  plt.ylabel('Transition probability')
  if plt_idx == 3:
    plt.xlabel('Distance to object (degrees)')
  plt.fill_between(x,ll,ul,alpha=.33)

def run_analysis(dataset: str, coding: str):

  num_objects = _get_num_objects(dataset)
  
  # Load frame-level data
  print('Loading frame-level data...')
  df = create_subjects_csvs.get_frame_data(coding=coding, dataset=dataset)

  # Compute next object for each pair of consecutive frames.
  df = stats_utils.add_next_object_column(df)
  
  # For computational reasons, we subsample non-transitions. Later we scale the
  # regression response to account for this.
  subsample_proportion = 0.001
  if coding == 'HMM':
    subsample_proportion = 1
  
  print('Computing frame-wise transition data...')
  distances_dict = {'distance': [], 'is_transition': [], 'is_from_target': [],
                    'is_to_target': [], 'subject_id': [], 'age': []}
  for idx, row in df.iterrows():
    source_x = row[f'object_{row["HMM"]}_x']
    source_y = row[f'object_{row["HMM"]}_y']
    for object_idx in range(num_objects):
      if object_idx != row['HMM']:
        is_transition = (row['next_frame_HMM'] == object_idx)
        if not is_transition:
          if random.random() > subsample_proportion:
            continue
        destination_x = row[f'object_{object_idx}_x']
        destination_y = row[f'object_{object_idx}_y']
        distance = math.sqrt((source_x - destination_x)**2
                           + (source_y - destination_y)**2)
        distances_dict['distance'].append(distance)
        distances_dict['is_transition'].append(is_transition)
        distances_dict['is_from_target'].append(row['HMM'] == 0)
        distances_dict['is_to_target'].append(row['next_frame_HMM'] == 0)
        distances_dict['subject_id'].append(row['subject_id'])
        distances_dict['age'].append(row['age'])
  
  df = pd.DataFrame(distances_dict)
  df['is_transition_from_target'] = (df['is_transition'] & df['is_from_target'])
  df['is_transition_to_target'] = (df['is_transition'] & df['is_to_target'])
  df['is_transition_to_distractor'] = (df['is_transition'] & ~df['is_from_target'] & ~df['is_to_target'])
  df['distance_in_degrees'] = df['distance'].map(stats_utils.pixels_to_degrees)

  # Correct for the fact that we downsampled non-transitions by 1000
  df['is_transition'] *= subsample_proportion
  df['is_transition_from_target'] *= subsample_proportion
  df['is_transition_to_target'] *= subsample_proportion
  df['is_transition_to_distractor'] *= subsample_proportion

  stats_utils.report_ttest_1sample(
      null_hypothesis="mean(effect of distance on transitions from target) == 0",
      sample=_get_corrs_by_subject(df, 'is_transition_from_target'),
      popmean=0)
  stats_utils.report_ttest_1sample(
      null_hypothesis="mean(effect of distance on transitions to target) == 0",
      sample=_get_corrs_by_subject(df, 'is_transition_to_target'),
      popmean=0)
  stats_utils.report_ttest_1sample(
      null_hypothesis="mean(effect of distance on transitions to distractors) == 0",
      sample=_get_corrs_by_subject(df, 'is_transition_to_distractor'),
      popmean=0)

  stats_utils.linreg_summary_and_plot(
      x='age',
      y='is_transition_from_target',
      data=_get_corrs_by_age(df, 'is_transition_from_target').reset_index().dropna(),
      name='Location Accuracy over Age')
  stats_utils.linreg_summary_and_plot(
      x='age',
      y='is_transition_to_target',
      data=_get_corrs_by_age(df, 'is_transition_to_target').reset_index().dropna(),
      name='Location Accuracy over Age')
  stats_utils.linreg_summary_and_plot(
      x='age',
      y='is_transition_to_distractor',
      data=_get_corrs_by_age(df, 'is_transition_to_distractor').reset_index().dropna(),
      name='Location Accuracy over Age')

  print('Plotting transition probabilities over distance...')
  plt.figure(figsize=(5, 8))
  df_from_target = df[df['is_from_target']].reset_index()
  _plot_loess(x=df_from_target['distance_in_degrees'],
             y=df_from_target['is_transition_from_target'],
             plt_idx=1,
             dataset=dataset,
             subsample_proportion=subsample_proportion)
  plt.title(r'Target $\to$ Distractor')
  
  df_from_distractor = df[~df['is_from_target']].reset_index()
  _plot_loess(x=df_from_distractor['distance_in_degrees'],
             y=df_from_distractor['is_transition_to_target'],
             plt_idx=2,
             dataset=dataset,
             subsample_proportion=subsample_proportion)
  plt.title(r'Distractor $\to$ Target')
  
  _plot_loess(x=df_from_distractor['distance_in_degrees'],
             y=df_from_distractor['is_transition_to_distractor'],
             plt_idx=3,
             dataset=dataset,
             subsample_proportion=subsample_proportion)
  plt.title(r'Distractor $\to$ Distractor')
  
  plt.subplots_adjust(left=0.2, right=0.94, top=0.95, bottom=0.08, hspace=0.3)
  
  plt.show()

if __name__ == '__main__':
  run_analysis('ORIGINAL', 'HMM')
