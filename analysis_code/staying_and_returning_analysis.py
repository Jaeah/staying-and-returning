"""This module implements analyses comparing staying and returning."""
import itertools
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
sns.set()
np.set_printoptions(suppress=True)

import create_subjects_csvs
import stats_utils
import effect_of_distance_analysis

_DATASET = 'ORIGINAL'
_CODING = 'HMM'

def report_statistics_and_make_plots():
  """Generates statistics and plots reported in paper."""

  df = create_subjects_csvs.get_experiment_data(coding=_CODING, dataset=_DATASET)

  # Basic statistics about each value
  stats_utils.report_univariate_statistics(name='Location Accuracy',
                                           sample=df['loc_acc'])
  stats_utils.report_univariate_statistics(name='Memory Accuracy',
                                           sample=df['mem_check'])
  stats_utils.report_univariate_statistics(name='PTDT',
                                           sample=df['returning'])
  stats_utils.report_univariate_statistics(name='NDT',
                                           sample=df['staying'])
  stats_utils.report_univariate_statistics(name='NDTDF',
                                           sample=df['staying_df'])

  # Compare statistics to chance values
  stats_utils.report_ttest_1sample(null_hypothesis="mean(Loc Acc) == 1/36",
                                   sample=df['loc_acc'], popmean=1/36)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(Mem Acc) == 1/4",
                                   sample=df['mem_check'], popmean=1/4)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(PTDT) == 1/6",
                                   sample=df['returning'], popmean=1/6)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(NDT) == 0",
                                   sample=df['staying'], popmean=0)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(NDTDF) == 0",
                                   sample=df['staying_df'], popmean=0)
  stats_utils.report_ttest_1sample(null_hypothesis="mean(NDT - NDTDF) == 0",
                                   sample=df['staying'] - df['staying_df'], popmean=0)

  df2 = df[['age', 'loc_acc', 'mem_check', 'returning', 'staying']].dropna()._get_numeric_data()
  dfcols = pd.DataFrame(columns=df2.columns)
  pvalues = dfcols.transpose().join(dfcols, how='outer')
  for r in df2.columns:
      for c in df2.columns:
          pvalues[r][c] = round(stats.pearsonr(df2[r], df2[c])[1], 4)
  print('Pearson correlations between performance measures:')
  print(df2.corr())
  print('p-values for correlations between performance measures:')
  print(pvalues)

  # Linearly regress statistics over age
  stats_utils.linreg_summary_and_plot(x='age', y='loc_acc', data=df, name='Location Accuracy over Age', plot=False)
  stats_utils.linreg_summary_and_plot(x='age', y='mem_check', data=df, name='Memory Accuracy over Age', plot=False)
  stats_utils.linreg_summary_and_plot(x='mem_check', y='loc_acc', data=df, name='Location Accuracy over Memory Accuracy', plot=False)
  stats_utils.linreg_summary_and_plot(x='returning', y='staying', data=df, name='NDT over PTDT', plot=False)

  # Linearly regress statistics over age
  plt.subplot(2, 2, 1)
  plt.xlim((3.5, 6))
  plt.ylim((0, 1))
  plt.plot([3.5, 6], [1/6, 1/6], c='red', ls='--')
  stats_utils.linreg_summary_and_plot(x='age', y='returning', data=df, name='PTDT over age')
  plt.xlabel('Age (years)')
  plt.ylabel('PTDT (proportion)')

  plt.subplot(2, 2, 3)
  plt.xlim((3.5, 6))
  plt.ylim((-0.4, 2.4))
  plt.plot([3.5, 6], [0, 0], c='red', ls='--')
  stats_utils.linreg_summary_and_plot(x='age', y='staying', data=df, name='NDT over age')
  plt.xlabel('Age (years)')
  plt.ylabel('NDT (seconds)')

  plt.subplot(2, 2, 2)
  plt.xlim((0, 1))
  plt.ylim((0, 1))
  plt.plot([0, 1], [1/36, 1/36], c='red', ls='--')
  plt.plot([1/6, 1/6], [0, 1], c='red', ls='--')
  stats_utils.linreg_summary_and_plot(x='returning', y='loc_acc', data=df,
                                      name='location accuracy over PTDT')
  plt.xlabel('PTDT (proportion)')
  plt.ylabel('Location Accuracy')

  plt.subplot(2, 2, 4)
  plt.xlim((-0.4, 2.4))
  plt.ylim((0, 1))
  plt.plot([-1, 3], [1/36, 1/36], c='red', ls='--')
  plt.plot([0, 0], [0, 1], c='red', ls='--')
  stats_utils.linreg_summary_and_plot(x='staying', y='loc_acc', data=df,
                                      name='location accuracy over NDT')
  plt.xlabel('NDT (seconds)')
  plt.ylabel('Location Accuracy')
  plt.tight_layout()

  # Mediation Analysis
  stats_utils.mediation_analysis(x='age', y='loc_acc', m='returning', data=df,
                                 title='PTDT mediating effect of Age on Loc Acc')

  # Differential regression of NDT ranks and PTDT ranks
  ranks = np.concatenate((stats.rankdata(df['returning']), stats.rankdata(df['staying'])))
  x_age = np.concatenate((df['age'], df['age']))
  measure_type = np.concatenate((np.zeros_like(df['returning']),
                                  np.ones_like(df['staying'])))
  x_interaction = np.multiply(x_age, measure_type)
  X = np.stack((x_age, measure_type, x_interaction), axis=1)
  print(sm.OLS(ranks, sm.add_constant(X)).fit().summary())
  ranks_df = pd.DataFrame(data={'Age': x_age, 'Rank': ranks, 'Measure': measure_type})
  ranks_df['Measure'] = ranks_df['Measure'].map({0.0: 'Returning', 1.0: 'Staying'})
  lmplot = sns.lmplot(x='Age', y='Rank', hue='Measure', data=ranks_df)
  lmplot._legend.remove()
  plt.legend(loc='lower right')
  plt.xlabel('Age (years)')
  plt.tight_layout()

  effect_of_distance_analysis.run_analysis(_DATASET, _CODING)

  plt.show()

def main():
  """Performs and reports analyses comparing staying and returning."""
  report_statistics_and_make_plots()

if __name__ == '__main__':
  main()
