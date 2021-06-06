"""Statistical helper functions for staying_vs_returning.py."""

from typing import Iterable

from datetime import datetime
from enum import Enum, auto
import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
import pandas as pd

from typing import Callable, Collection, List, NamedTuple, Tuple

_TRIALS_TO_KEEP = list(range(1, 11))

Run = NamedTuple('Run', [('object', int), ('length', int)])

def pixels_to_degrees(d: float):
  """Converts a distance in pixels to a distance in degrees of visual field.
  
  For simplicity, this function assumes:
    1) The distance is centered at the center of the visual field. This
       assumption is fairly mild, since the TrackIt display is centered and
       takes up only a small portion of the visual field.
    2) The distance is horizontal. This introduces an error of <1% due to
       slight differences in measuring the horizontal and vertical screen sizes.
  """
  pixel_width = 34.2/1920  # Width of a pixel in centimeters.
  display_distance = 50  # Distance of participant from display in centimeters.
  return math.degrees(2*math.atan((d/2)*pixel_width/display_distance))
  

def calc_age(experiment):

  birthdate = datetime.strptime(
      experiment.datatypes['trackit'].metadata['Birthdate'], '%m/%d/%Y')
  if birthdate.year < 2000:
    # Fix birthdates that were miscoded with year 00YY instead of 20YY
    birthdate = birthdate.replace(year=birthdate.year+2000)

  test_date = datetime.strptime(
      experiment.datatypes['trackit'].metadata['Test Date'], '%m/%d/%Y')

  age = (test_date - birthdate).days/365.25
  return age

def add_next_object_column(df: pd.DataFrame) -> pd.DataFrame:
  """Compute next object for each pair of consecutive frames."""

  trial_dfs = []
  for key, trial_df in df.groupby(by=['subject_id', 'condition', 'trial_num']):

    # Suppress SettingWithCopyWarning because we will overwrite df later
    trial_df = trial_df.copy()

    trial_df.sort_values(by='frame', inplace=True)
    trial_df['next_frame_HMM'] = trial_df['HMM'].shift(-1)
    trial_dfs.append(trial_df)

  df = pd.concat(trial_dfs, ignore_index=True)

  # Remove "transitions" to End-Of-Trial
  df = df[~df['next_frame_HMM'].isnull()]
  df['next_frame_HMM'] = df['next_frame_HMM'].astype(int)
  
  # Remove "transitions" to/from Off-Screen
  df = df[(df['HMM'] >= 0) & (df['next_frame_HMM'] >= 0)]

  return df


def average_over_trials(metric: Callable, experiment):
  """Computes the average of a metric over trial."""
  return np.nanmean(
          [metric(experiment.datatypes['eyetrack'].trials[trial_idx])
           for trial_idx in _TRIALS_TO_KEEP])

def experiment_loc_acc(experiment):
  return np.mean(
      [(experiment
        .datatypes['trackit']
        .trials[trial_idx]
        .trial_metadata['gridClickCorrect'] == 'true')
       for trial_idx in _TRIALS_TO_KEEP])

def trial_ptdt(trial, omit_missing_frames=True):
  """Computes proportion of transitions from distractors to target (PTDT)."""
  frames = trial.HMM_MLE
  if omit_missing_frames:
    frames = frames[frames >= 0]

  transitions_from_distractor_to_target = 0
  transitions_from_distractor = 0
  for first, second in zip(frames, frames[1:]):
    if first > 0 and second != first and second >= 0:
      transitions_from_distractor += 1
      if second == 0:
        transitions_from_distractor_to_target += 1
  try:
    return transitions_from_distractor_to_target \
            /transitions_from_distractor
  except ZeroDivisionError:
    return float('nan')

def experiment_ptdt(experiment, omit_missing_frames=True) -> float:
  return average_over_trials(trial_ptdt, experiment)

def calc_run_lengths(sequence: List[int]) -> List[Run]:
  """Computes lengths of contiguous runs of the same object."""
  return [Run(object=g[0], length=len(list(g[1])))
          for g in itertools.groupby(sequence)]


def trial_ndt(trial, omit_missing_frames=True, drop_first=False, coding='HMM'):
  """Computes normalized duration on target (NDT) in seconds."""
  frames = trial.HMM_MLE

  group_lengths = [(g[0], len(list(g[1]))) for g in itertools.groupby(frames)]

  if omit_missing_frames:
    group_lengths = [l for l in group_lengths if l[0] >= 0]

  if drop_first:
    group_lengths = group_lengths[1:]

  mean_on_target_group_length = np.mean(
      [l[1] for l in group_lengths if l[0] == 0])
  mean_nonmissing_group_length = np.mean(
      [l[1] for l in group_lengths])

  fps = 60 if (coding == 'HMM') else 10

  return (mean_on_target_group_length - mean_nonmissing_group_length)/fps

def experiment_ndt(experiment, omit_missing_frames=True) -> float:
  return average_over_trials(trial_ndt, experiment)

def trial_pfot(trial, omit_missing_frames=True):
  """Computes proportion of frames on target (PFT)."""
  frames = trial.HMM_MLE
  if omit_missing_frames:
    frames = frames[frames >= 0]
  return np.mean(frames == 0)

def experiment_pfot(experiment, omit_missing_frames=True) -> float:
  return average_over_trials(trial_pfot, experiment)

def trial_atd(trial, omit_missing_frames=True):
  """Computes average tracking duration (ATD) in seconds."""
  frames = trial.HMM_MLE
  if omit_missing_frames:
    frames = frames[frames >= 0]
  total_frames = len(frames)
  num_runs = len([run for run in calc_run_lengths(frames)])
  if num_runs == 0:
    return float('nan')
  return (total_frames/num_runs)/60

def experiment_atd(experiment, omit_missing_frames=True) -> float:
  return average_over_trials(trial_atd, experiment)

def trial_atr(trial, omit_missing_frames=True):
  """Computes average time to return (ATR) in seconds."""
  frames = trial.HMM_MLE
  if omit_missing_frames:
    frames = frames[frames >= 0]

  runs = calc_run_lengths(trial.HMM_MLE)
  return_times = []
  current_return_time = 0
  for run in runs:
    if run.object == 0:
      return_times.append(current_return_time/60)
      current_return_time = 0
    else:
      current_return_time += run.length
  return np.mean(return_times)

def experiment_atr(experiment, omit_missing_frames=True) -> float:
  return average_over_trials(trial_atr, experiment)

def trial_wtd(trial, omit_missing_frames=True):
  """Computes within-trial decrement (WDT)."""
  x = np.arange(len(trial.HMM_MLE))/60
  y = (trial.HMM_MLE == 0)
  if omit_missing_frames:
    x = x[trial.HMM_MLE >= 0]
    y = y[trial.HMM_MLE >= 0]
  return linear_regression_with_CIs(x/60, y, return_CIs=False)

def experiment_wtd(experiment,
                   omit_missing_frames=True) -> float:
  return average_over_trials(trial_wtd, experiment,
                             omit_missing_frames=omit_missing_frames)

def experiment_btd(experiment, omit_missing_frames=True) -> float:
  """Computes between-trials decrement (BTD).
  
  Note that, unlike the other performance metrics, BTD can only be computed at
  the experiment level, not at the trial level.
  """
  trial_pfots = []
  for trial_idx in _TRIALS_TO_KEEP:
    frames = experiment.datatypes['eyetrack'].trials[trial_idx].HMM_MLE
    if omit_missing_frames:
      frames = frames[frames >= 0]
    trial_pfots.append(np.mean(frames == 0))
  zipped = [(x, y) for (x, y) in zip(_TRIALS_TO_KEEP, trial_pfots)
            if not math.isnan(y)]
  xs, ys = zip(*zipped)
  slope = linear_regression_with_CIs(xs, ys, return_CIs=False)
  return slope

def report_univariate_statistics(
      name: str, sample: List[float], alpha: float = 0.05):
  """Pretty prints basic statistics about a univariate distribution."""
  mean = np.mean(sample)
  std = np.std(sample)
  n = len(sample)
  lower = mean - 1.96*std/math.sqrt(n)
  upper = mean + 1.96*std/math.sqrt(n)
  print(f'Distribution Statistics for {name}')
  print(f'Sample mean: {mean:.2f} ({lower:.2f}, {upper:.2f}), Sample SD: {std:.2f}\n')

def report_ttest_1sample(
        null_hypothesis: str, sample: List[float], popmean: float,
        one_sided: bool = False, alpha: float = 0.05):
  """Performs and pretty-prints results of a one-sample t-test.
  
  By default, the test is two-sided.
  If one-sided is True, the test is assumed to be for sample_mean > popmean.

  Args:
    null_hypothesis: string describing null hypothesis being tested
    sample: data sample
    popmean: population mean under null hypothesis
    one_sided: If True, perform a 1-sided test; else, perform a 2-sided test
    alpha: Test level
  """

  t_value, p_value = stats.ttest_1samp(sample, popmean)
  if one_sided and t_value > 0:
      p_value /= 2
  print('Test for null hypothesis "{}".'.format(null_hypothesis))
  print('Sample mean: {}, Sample SD: {}'.format(np.mean(sample), np.std(sample)))
  print('t({})={}, p={}.'.format(len(sample)-1, t_value, p_value))
  if p_value < alpha:
    print('Reject null hypothesis.\n')
  else:
    print('Fail to reject null hypothesis.\n')

def report_ttest_2sample(null_hypothesis, sample1, sample2, paired, alpha=0.05):
  """Pretty-prints results of a two-sided two-sample t-test."""

  if paired:
    t_value, p_value = stats.ttest_rel(sample1, sample2)
  else:
    t_value, p_value = stats.ttest_ind(sample1, sample2)
  print('Test for null hypothesis "{}".'.format(null_hypothesis))
  print('Sample 1 mean: {}, Sample 1 SD: {}'.format(np.mean(sample1), np.std(sample1)))
  print('Sample 2 mean: {}, Sample 2 SD: {}'.format(np.mean(sample2), np.std(sample2)))
  print('t({})={}, p={}.'.format(len(sample1)-1, t_value, p_value))
  if p_value < alpha:
    print('Reject null hypothesis.\n')
  else:
    print('Fail to reject null hypothesis.\n')

def linreg_summary_and_plot(x: str, y: str, data: pd.DataFrame, name=None, plot=True):

  if name:
    print('\n{}:\n'.format(name.upper()))

  xs = data[x]
  ys = data[y]
  print(sm.OLS(ys, sm.add_constant(xs)).fit().summary())
  if plot:
    sns.regplot(xs, ys, truncate=False)

def linear_regression_with_CIs(x, y, return_CIs = True):
  if len(x) < 2:
    if return_CIs:
      return float('nan'), float('nan'), float('nan')
    return float('nan')

  model = sm.OLS(y, sm.add_constant(x)).fit()
  if not return_CIs:
    return model.params[1]
  CI = model.conf_int(alpha=0.05, cols=[1])[0]
  return model.params[1], CI[0], CI[1]

def _calc_indirect_effect(x, y, m):
  """Estimate standardized indirect effect and proportion of mediation.
  
  Specifically, computes the estimated indirect effect and proportion of the
  relationship of x on y that is mediated by m.
  
  Args:
    x: samples from 
  """
  x = stats.zscore(x)
  y = stats.zscore(y)
  m = stats.zscore(m)
  total_effect = sm.OLS(y, sm.add_constant(x)).fit().params[1]
  xs = np.stack((x, m), axis=1)
  direct_effect = sm.OLS(y, sm.add_constant(xs)).fit().params[1]
  indirect_effect = total_effect - direct_effect
  proportion_mediated = indirect_effect/total_effect
  return indirect_effect, proportion_mediated

def mediation_analysis(x: str, y: str, m: str, data: pd.DataFrame, title, num_reps = 10000):

  conditions = data['condition'].unique()
  if len(conditions) != 1:
    raise ValueError('Trying to regress multiple conditions at once: '
                     f'{conditions}')
  condition = conditions[0]

  indirect_effect, prop_mediated = _calc_indirect_effect(data[x], data[y], data[m])
  subsampled_indirect_effects = np.zeros((num_reps,))
  subsampled_prop_mediated = np.zeros((num_reps,))
  for rep in range(num_reps):
    samples = random.choices(range(len(data[x])), k=len(data[x]))
    x_sub = [data.loc[data.index[i], x] for i in samples]
    y_sub = [data.loc[data.index[i], y] for i in samples]
    m_sub = [data.loc[data.index[i], m] for i in samples]
    subsampled_indirect_effects[rep], subsampled_prop_mediated[rep] = _calc_indirect_effect(x_sub, y_sub, m_sub)

  indirect_effect_CI_lower = np.percentile(subsampled_indirect_effects, 2.5)
  indirect_effect_CI_upper = np.percentile(subsampled_indirect_effects, 97.5)
  p_value = np.mean(subsampled_indirect_effects < 0)
  prop_mediated_CI_lower = np.percentile(subsampled_prop_mediated, 2.5)
  prop_mediated_CI_upper = np.percentile(subsampled_prop_mediated, 97.5)
  print(title)
  print(f'Indirect Effect: {indirect_effect:.3f}    95% CI: ({indirect_effect_CI_lower:.3f}, {indirect_effect_CI_upper:.3f})')
  print(f'Proportion Mediation: {prop_mediated:.3f}    95% CI: ({prop_mediated_CI_lower:.3f}, {prop_mediated_CI_upper:.3f})')
  print(f'p-value: {p_value:.3f}')
