import numpy as np
import pandas as pd
import pickle

import stats_utils

def trials(dataset: str):
  """Returns a list of non-practice trial numbers in the given dataset."""
  if dataset == 'ORIGINAL':
    return list(range(1, 11))
  if dataset == 'LABELING':
    return list(range(1, 13))
  raise ValueError(f'dataset should be \'ORIGINAL\' or \'LABELING\' '
                   f'but was {dataset}')


def condition(dataset: str):
  """Returns the condition to be used from the given dataset."""
  if dataset == 'ORIGINAL':
    return 'noshrinky'
  if dataset == 'LABELING':
    return 'labeled'
  raise ValueError(f'dataset should be \'ORIGINAL\' or \'LABELING\' '
                   f'but was {dataset}')


def _load_subjects(dataset: str, coding: str):
  """Load and filter subjects from pickle file."""
  fname = f'{dataset.lower()}_300.pickle'
  print('Loading data from file ' + fname + '...')
  with open(fname, 'rb') as input_file:
    subjects = pickle.load(input_file, encoding='latin1')
  subjects = [subject for subject in subjects.values()
              if subject_is_good(subject, dataset=dataset)]

  if coding == 'HUMAN':
    if dataset != 'ORIGINAL':
      raise ValueError('HUMAN coding is only available for ORIGINAL dataset, '
                       f'but dataset was {dataset}')
    _replace_hmm_with_human_coded(subjects)
  elif coding != 'HMM':
    raise ValueError(f'coding must be \'HMM\' or \'HUMAN\' but was {coding}')

  print('Loaded {} good subjects.\n'.format(len(subjects)))
  return subjects


def subject_is_good(subject, dataset: str ='ORIGINAL'):
  """Whether a subject satisfies inclusion criteria."""

  num_valid_trials_needed = len(trials(dataset))/2
  all_experiments_have_enough_trials = all(
      [len(x.trials_to_keep) >= num_valid_trials_needed for x in subject.experiments.values()]
  )
  if not all_experiments_have_enough_trials:
    return False

  condition_to_use = condition(dataset)
  if condition_to_use not in subject.experiments:
    return False
  subject.experiments = {condition_to_use: subject.experiments[condition_to_use]}

  if dataset == 'LABELING':
    experiment = subject.experiments[condition_to_use]
    # LABELING dataset experiments do not have age precomputed, so add this
    experiment.age = stats_utils.calc_age(experiment)
    # Discard some participants in the LABELING dataset with erroneous ages
    if 3 > experiment.age or 5.8 < experiment.age:
      return False

  return True

def _replace_hmm_with_human_coded(subjects):
  print('Replacing HMM with human coding...')
  coding = {
      'Obect 0': 0,  # 'Obect 0' is a typo appearing in part of the original data.
      'Object 0': 0,
      'Object 1': 1,
      'Object 2': 2,
      'Object 3': 3,
      'Object 4': 4,
      'Object 5': 5,
      'Object 6': 6,
      # Code both Off Task and Off Screen as missing data.
      'Off Task': -1,
      'Off Screen': -1,
      float('nan'): -1
  }
  for subject in subjects:
    for experiment in subject.experiments.values():
      if experiment.ID == 'shrinky':
        continue
      for trial_idx in trials('ORIGINAL'):
        file_name = f'{subject.ID}_{experiment.ID}_trial_{trial_idx}_coding.csv'

        # Try loading from Coder 1's data; if not found, try Coder 3's data.
        try:
          df = pd.read_csv(f'../human_coded/coder1/{file_name}')
        except FileNotFoundError:
          df = pd.read_csv(f'../human_coded/coder3/{file_name}')

        human_coding = df['Default'].map(coding)
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]
        eyetrack_trial.HMM_MLE = human_coding.to_numpy()

def get_frame_data(dataset: str = 'ORIGINAL', coding: str = 'HMM') -> pd.DataFrame:

  subjects = _load_subjects(dataset=dataset, coding=coding)
  if coding == 'HUMAN':
    if dataset != 'ORIGINAL':
      raise ValueError('HUMAN coding is only available for ORIGINAL dataset, '
                       f'but dataset was {dataset}')
    _replace_hmm_with_human_coded(subjects)
  elif coding != 'HMM':
    raise ValueError(f'coding must be \'HMM\' or \'HUMAN\' but was {coding}')

  table_as_dict = {
      'subject_id': [], 'age': [], 'condition': [], 'trial_num': [], 'target': [],
      'trial_len': [], 'loc_acc': [], 'error_type': [], 'frame': [], 'HMM': [],
      'shape': []
  }

  for subject in subjects:
    for experiment in subject.experiments.values():
      for trial_idx in trials(dataset):
        trackit_trial = experiment.datatypes['trackit'].trials[trial_idx]
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]
        object_names = trackit_trial.meta_data['object_names']
        for frame, HMM in enumerate(eyetrack_trial.HMM_MLE):

          # Experiment-level data
          table_as_dict['subject_id'].append(subject.ID)
          table_as_dict['condition'].append(experiment.ID)
          table_as_dict['age'].append(experiment.age)

          # Trial-level data
          table_as_dict['trial_num'].append(trial_idx)
          table_as_dict['target'].append(trackit_trial.trial_metadata['target'])
          table_as_dict['trial_len'].append(len(eyetrack_trial.HMM_MLE))
          table_as_dict['loc_acc'].append(
              trackit_trial.trial_metadata['gridClickCorrect'] == 'true')
          table_as_dict['error_type'].append(
              trackit_trial.trial_metadata['errorType'])

          # Frame-level data
          table_as_dict['frame'].append(frame)
          table_as_dict['HMM'].append(HMM)
          table_as_dict['shape'].append(object_names[HMM])

        # When coding=='HUMAN', we need to downsample the temporal resolution of
        # the TrackIt object locations by 6.
        if coding == 'HUMAN':
          step = 6
        else:
          step = 1

        # Frame-level array data
        for object_idx in range(trackit_trial.object_positions.shape[0]):
          object_x_col_name = f'object_{object_idx}_x'
          object_y_col_name = f'object_{object_idx}_y'
          if not object_x_col_name in table_as_dict:
            table_as_dict[object_x_col_name] = []
            table_as_dict[object_y_col_name] = []
          x_coords = trackit_trial.object_positions[object_idx, ::step, 0]
          y_coords = trackit_trial.object_positions[object_idx, ::step, 1]

          # Since the human coding is not perfectly aligned with the HMM coding,
          # we may need to drop 1 or 2 samples at the end.
          min_len = min(len(eyetrack_trial.HMM_MLE), len(x_coords))
          x_coords = x_coords[:min_len]
          y_coords = y_coords[:min_len]

          table_as_dict[object_x_col_name].extend(list(x_coords))
          table_as_dict[object_y_col_name].extend(list(y_coords))

  return pd.DataFrame(table_as_dict)
  

def get_trial_data(dataset: str = 'ORIGINAL', coding: str = 'HMM') -> pd.DataFrame:
  subjects = _load_subjects(dataset=dataset, coding=coding)

  table_as_dict = {'subject_id': [], 'age': [], 'condition': [], 'target': [],
                   'trial_num': [], 'trial_len': [], 'loc_acc': [],
                   'mem_check': [], 'pfot': [], 'returning': [], 'staying': [],
                   'proportion_missing_eyetracking': [], 'atr': [], 'wtd': [],
                   'atd': [], 'staying_df': []}

  for subject in subjects:
    for experiment in subject.experiments.values():
      for trial_idx in trials(dataset):
        trackit_trial = experiment.datatypes['trackit'].trials[trial_idx]
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]

        # Experiment-level data
        table_as_dict['subject_id'].append(subject.ID)
        table_as_dict['condition'].append(experiment.ID)
        table_as_dict['age'].append(experiment.age)

        # Trial-level data
        table_as_dict['trial_num'].append(trial_idx)
        table_as_dict['trial_len'].append(len(eyetrack_trial.HMM_MLE))
        table_as_dict['target'].append(trackit_trial.trial_metadata['target'])
        table_as_dict['loc_acc'].append(
            trackit_trial.trial_metadata['gridClickCorrect'] == 'true')
        table_as_dict['mem_check'].append(
            trackit_trial.trial_metadata['lineupClickCorrect'] == 'true')
        table_as_dict['pfot'].append(stats_utils.trial_pfot(eyetrack_trial))
        table_as_dict['returning'].append(stats_utils.trial_ptdt(eyetrack_trial))
        table_as_dict['staying'].append(stats_utils.trial_ndt(eyetrack_trial, coding=coding))
        table_as_dict['staying_df'].append(stats_utils.trial_ndt(eyetrack_trial, drop_first=True, coding=coding))
        table_as_dict['atd'].append(stats_utils.trial_atd(eyetrack_trial))
        table_as_dict['wtd'].append(stats_utils.trial_wtd(eyetrack_trial))
        table_as_dict['atr'].append(stats_utils.trial_atr(eyetrack_trial))
        table_as_dict['proportion_missing_eyetracking'].append(
            eyetrack_trial.proportion_missing)

  return pd.DataFrame(table_as_dict)
  

def get_experiment_data(
    memcheck_correct_only: bool = False,
    correct_trials_only: bool = False,
    dataset: str = 'ORIGINAL',
    coding: str = 'HMM',
) -> pd.DataFrame:
  subjects = _load_subjects(dataset=dataset, coding=coding)

  table_as_dict = {
      'subject_id': [], 'sex': [], 'age': [], 'condition': [], 'btd': [],
      'loc_acc': [], 'mem_check': [], 'pfot': [], 'returning': [],
      'staying': [], 'proportion_missing_eyetracking': [], 'atr': [],
      'wtd': [], 'atd': [], 'staying_df': []
  }

  for subject in subjects:
    for experiment in subject.experiments.values():
      for trial_idx in trials(dataset):
        trackit_trial = experiment.datatypes['trackit'].trials[trial_idx]
        eyetrack_trial = experiment.datatypes['eyetrack'].trials[trial_idx]

        # Experiment-level data
        table_as_dict['subject_id'].append(subject.ID)
        table_as_dict['sex'].append(experiment.datatypes['trackit'].metadata['Gender'])
        table_as_dict['condition'].append(experiment.ID)
        table_as_dict['age'].append(experiment.age)
        table_as_dict['btd'].append(stats_utils.experiment_btd(experiment))

        # Trial-level data
        table_as_dict['loc_acc'].append(
            trackit_trial.trial_metadata['gridClickCorrect'] == 'true')
        table_as_dict['mem_check'].append(
            trackit_trial.trial_metadata['lineupClickCorrect'] == 'true')
        table_as_dict['pfot'].append(stats_utils.trial_pfot(eyetrack_trial))
        table_as_dict['returning'].append(stats_utils.trial_ptdt(eyetrack_trial))
        table_as_dict['staying'].append(stats_utils.trial_ndt(eyetrack_trial, coding=coding))
        table_as_dict['staying_df'].append(stats_utils.trial_ndt(eyetrack_trial, drop_first=True, coding=coding))
        table_as_dict['atd'].append(stats_utils.trial_atd(eyetrack_trial))
        table_as_dict['wtd'].append(stats_utils.trial_wtd(eyetrack_trial))
        table_as_dict['atr'].append(stats_utils.trial_atr(eyetrack_trial))
        table_as_dict['proportion_missing_eyetracking'].append(
            eyetrack_trial.proportion_missing)

  df = pd.DataFrame(table_as_dict)
  if correct_trials_only:
    df = df[df['loc_acc']]
  if memcheck_correct_only:
    df = df[df['mem_check']]

  return (df.groupby(by=['subject_id', 'sex', 'condition'])
            .mean()
            .reset_index())
  

def main():
  # print(get_frame_data())
  # print(get_trial_data())
  print(get_experiment_data(coding='HMM', dataset='ORIGINAL'))
  

if __name__ == '__main__':
  main()
