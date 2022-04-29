"""This module analyzes the effect of Target familiarity on performance."""
import create_subjects_csvs
import stats_utils

_DATASET = 'ORIGINAL'
_CODING = 'HMM'

def main():
  """Generates statistics and plots reported in paper."""

  df = create_subjects_csvs.get_trial_data(coding=_CODING, dataset=_DATASET)

  df['is_familiar'] = df['target'].apply(
      lambda target: target in {'Box', 'Triangle', 'Circle'})

  stats_utils.report_ttest_2sample('mean(Location Accuracy, Familiar) == mean(Location Accuracy, Unfamiliar)',
                                   df['loc_acc'][df['is_familiar']],
                                   df['loc_acc'][~df['is_familiar']],
                                   paired=False)
  stats_utils.report_ttest_2sample('mean(Memory Accuracy, Familiar) == mean(Memory Accuracy, Unfamiliar)',
                                   df['mem_check'][df['is_familiar']],
                                   df['mem_check'][~df['is_familiar']],
                                   paired=False)
  stats_utils.report_ttest_2sample('mean(Returning, Familiar) == mean(Returning, Unfamiliar)',
                                   df['returning'][df['is_familiar']].dropna(),
                                   df['returning'][~df['is_familiar']].dropna(),
                                   paired=False)
  stats_utils.report_ttest_2sample('mean(staying, Familiar) == mean(staying, Unfamiliar)',
                                   df['staying'][df['is_familiar']].dropna(),
                                   df['staying'][~df['is_familiar']].dropna(),
                                   paired=False)

if __name__ == '__main__':
  main()
