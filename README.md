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
1. Use `lrzip` to uncompress the data files: ```lrunzip \*.lrz```
2. Install necessary Python modules: ```python -m pip install -r requirements.txt```
3. Navigate to the code directory: ```cd analysis_code```
4. Run the analysis script: ```python staying_and_returning_analysis.py```
