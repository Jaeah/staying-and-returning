# Temporal Dynamics of Attention in Young Children
This repository contains Python code underlying analyses in the paper "Temporal Dynamics of Attention in Young Children".

The below instructions explain the steps needed to reproduce the analyses reported in the paper.
All necessary data files needed are included in the repository.
These instructions were tested on Ubuntu 16.04, but should be easy to adapt to other \*nix systems.

## Prerequisites:
1. You will need [Python 3.6+](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get).
2. Since GitHub has a maximum file size of 100MB, some of the data files have been compressed using [`lrzip`](http://manpages.ubuntu.com/manpages/bionic/man1/lrzip.1.html). You will need `lrunzip` to decompress these files. This can be installed by `apt-get install lrzip`.
3. You should probably initialize and activate a Python [virtual environment](https://docs.python.org/3/tutorial/venv.html).

## To reproduce the analyses:
1. Install necessary Python modules:
```
python -m pip install -r requirements.txt
```
2. Navigate to the code directory:
```
cd analysis_code
```
3. Use `lrzip` to uncompress the data files:
```
lrunzip \*.lrz
```
4. For the main analyses, run the analysis script:
```
python staying_and_returning_analysis.py
```
By default, this will run the analyses of Experiment 1 with gaze data coded by the hidden Markov model. To run the Labeling Dataset analyses of Experiment 2, change Line 16 of `staying_and_returning_analysis.py` from `_DATASET = 'ORIGINAL'` to `_DATASET = 'LABELING'`. To run the Human Coding analyses of Appendix B, change Line 17 of `staying_and_returning_analysis.py` from `_CODING = 'HMM'` to `_CODING = 'HUMAN'`.
5. For the analyses of the effect of distance of transition probabilitys, run the distance script
```
python effect_of_distance_analysis.py
```
As with `staying_and_returning_analysis.py`, this will run the analyses in Experiment 1, and the analyses of Experiment 2 and Appendix B can be run by changing Lines 15 and 16 of `effect_of_distance_analysis.py` to `_DATASET = 'LABELING'` or `_CODING = 'HUMAN'`, respectively.
