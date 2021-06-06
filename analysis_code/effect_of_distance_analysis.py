# TODO: This script could be integrated into staying_vs_returning_analysis.py.
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from skmisc.loess import loess
import seaborn as sns
import statsmodels.api as sm

import create_subjects_csvs
import stats_utils

_DATASET = 'LABELING'
_CODING = 'HMM'

# Number of TrackIt objects_
def get_num_objects(dataset: str):
  """Returns the number of objects used in each trial of the given dataset."""
  if dataset == 'ORIGINAL':
    return 7
  if dataset == 'LABELING':
    return 5
  raise ValueError(f'dataset should be \'ORIGINAL\' or \'LABELING\' '
                   f'but was {dataset}')

def get_max_distance(dataset: str):
  """Returns the maximum usable distance value in the given dataset."""
  if dataset == 'ORIGINAL':
    return 1000
  if dataset == 'LABELING':
    return 700
  raise ValueError(f'dataset should be \'ORIGINAL\' or \'LABELING\' '
                   f'but was {dataset}')

_NUM_OBJECTS = get_num_objects(_DATASET)

# Load frame-level data
print('Loading frame-level data...')
df = create_subjects_csvs.get_frame_data(coding=_CODING, dataset=_DATASET)

# Compute next object for each pair of consecutive frames.
df = stats_utils.add_next_object_column(df)

# For computational reasons, we subsample non-transitions. Later we scale the
# regression response to account for this.
subsample_proportion = 0.001

print('Computing frame-wise transition data...')
distances_dict = {'distance': [], 'is_transition': [], 'is_from_target': [], 'is_to_target': [], 'subject_id': []}
for idx, row in df.iterrows():
  source_x = row[f'object_{row["HMM"]}_x']
  source_y = row[f'object_{row["HMM"]}_y']
  for object_idx in range(_NUM_OBJECTS):
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

  # if idx > 10000: break  # FOR TESTING ONLY

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

# # Plot distributions of transition and non-transition inter-object distances
# plt.figure()
# sns.distplot(df['distance'][df['is_transition']], label='Transitions')
# sns.distplot(df['distance'][~df['is_transition']], label='Non-Transitions')
# plt.xlabel('Distance to object')
# plt.ylabel('Probability')
# plt.legend()

# # Generate figure of p-value over threshold
# print('Plotting p-value over distance threshold...')
# thresholds = range(50, get_max_distance(_DATASET), 10)
# def linear_regressions_over_threshold(transition_type):
#   betas = []
#   pvalues = []
#   for threshold in thresholds:
#     thresholded_df = df[df['distance'] < threshold]
#     y = thresholded_df[transition_type]
#     x = thresholded_df['distance']
#     model = sm.OLS(y, sm.add_constant(x)).fit()
#     betas.append(model.params[1])
#     pvalues.append(model.pvalues[1])
#   return betas, pvalues
# 
# betas_from_target, pvalues_from_target = linear_regressions_over_threshold('is_transition_from_target')
# betas_to_target, pvalues_to_target = linear_regressions_over_threshold('is_transition_to_target')
# betas_to_distractor, pvalues_to_distractor = linear_regressions_over_threshold('is_transition_to_distractor')
# 
# stats_utils.linreg_summary_and_plot(x='distance', y='is_transition_from_target', data=df, name='Probability of Transitions from Target over Distance', plot=False)
# stats_utils.linreg_summary_and_plot(x='distance', y='is_transition_to_target', data=df, name='Probability of Transitions to Target over Distance', plot=False)
# stats_utils.linreg_summary_and_plot(x='distance', y='is_transition_to_distractor', data=df, name='Probability of Transitions from Distractor to Distractor over Distance', plot=False)

def get_beta(df: pd.DataFrame, response: str):
  return sm.OLS(df[response], sm.add_constant(df['distance_in_degrees'])).fit().params[1]

def get_beta_with_CIs(df: pd.DataFrame, response: str, num_bootstrap_samples: int = 10000):
  subject_ids = list(df['subject_id'].unique())
  bootstrap_sample_size =  len(subject_ids)

  betas = []
  for _ in range(num_bootstrap_samples):
    subsample_ids = random.choices(subject_ids, k=bootstrap_sample_size)
    subsampled_df = df[df['subject_id'].isin(subsample_ids)]
    betas.append(get_beta(subsampled_df, response))

  estimate = get_beta(df, response)
  lower = np.percentile(betas, 2.5)
  upper = np.percentile(betas, 97.5)
  print(f'Estimated coefficient: {estimate} '
        f'(bootstrapped 95% CI: ({lower}, {upper}))')
  return estimate, lower, upper

print('Computing bootstrapped regression coefficients...')
beta_from_target, lower_from_target, upper_from_target = get_beta_with_CIs(df, 'is_transition_from_target')
beta_to_target, lower_to_target, upper_to_target = get_beta_with_CIs(df, 'is_transition_to_target')
beta_to_distractor, lower_to_distractor, upper_to_distractor = get_beta_with_CIs(df, 'is_transition_to_distractor')

# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(thresholds, betas_from_target)
# plt.plot([0, get_max_distance(_DATASET)], [0, 0], c='r', ls='--')
# plt.yscale('symlog')
# plt.subplot(3, 1, 2)
# plt.plot(thresholds, betas_to_target)
# plt.plot([0, get_max_distance(_DATASET)], [0, 0], c='r', ls='--')
# plt.xlabel('beta')
# plt.yscale('symlog')
# plt.subplot(3, 1, 3)
# plt.plot(thresholds, betas_to_distractor)
# plt.plot([0, get_max_distance(_DATASET)], [0, 0], c='r', ls='--')
# plt.yscale('symlog')
# 
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(thresholds, pvalues_from_target)
# plt.plot([0, get_max_distance(_DATASET)], [1, 1], c='r', ls='--')
# plt.yscale('log')
# plt.subplot(3, 1, 2)
# plt.plot(thresholds, pvalues_to_target)
# plt.plot([0, get_max_distance(_DATASET)], [1, 1], c='r', ls='--')
# plt.xlabel('p-value')
# plt.yscale('log')
# plt.subplot(3, 1, 3)
# plt.plot(thresholds, pvalues_to_distractor)
# plt.plot([0, get_max_distance(_DATASET)], [1, 1], c='r', ls='--')
# plt.yscale('log')

# Plot transition likelihood as a function of inter-object distance, separately
# for each transition type
import pylab as plt
print('Plotting transition probabilities over distance...')
plt.figure(figsize=(5, 8))
def plot_loess(x, y, plt_idx):

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
  plt.xlim(left=0, right=stats_utils.pixels_to_degrees(get_max_distance(_DATASET)))
  y_margin = subsample_proportion/20
  plt.ylim(bottom=-y_margin, top=subsample_proportion+y_margin)
  plt.ylabel('Transition probability')
  if plt_idx == 3:
    plt.xlabel('Distance to object (degrees)')
  plt.fill_between(x,ll,ul,alpha=.33)

# plot_loess(x=df['distance'], y=df['is_transition'], plt_idx=1)
# plt.title(r'Any object $\to$ Any object')

df_from_target = df[df['is_from_target']].reset_index()
plot_loess(x=df_from_target['distance_in_degrees'],
           y=df_from_target['is_transition_from_target'],
           plt_idx=1)
plt.title(r'Target $\to$ Distractor')

df_from_distractor = df[~df['is_from_target']].reset_index()
plot_loess(x=df_from_distractor['distance_in_degrees'],
           y=df_from_distractor['is_transition_to_target'],
           plt_idx=2)
plt.title(r'Distractor $\to$ Target')

plot_loess(x=df_from_distractor['distance_in_degrees'],
           y=df_from_distractor['is_transition_to_distractor'],
           plt_idx=3)
plt.title(r'Distractor $\to$ Distractor')

plt.subplots_adjust(left=0.2, right=0.94, top=0.95, bottom=0.08, hspace=0.3)

plt.show()