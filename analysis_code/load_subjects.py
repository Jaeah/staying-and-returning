import csv, json
import numpy as np
from datetime import datetime

class Subject:
  """The basic unit of TrackIt data is the subject. Pretty much all statistics should be computed at the subject level first, and then aggregated across subjects, since only subjects can really be assumed to be IID."""
  def __init__(self, ID):
    """The only universal attribute of a subject is a unique string identifier, subject_ID"""
    self.ID = ID
    self.experiments = {}
    print('Creating new subject ' + ID)

  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

class Experiment: # Should also contain subject age at experiment time, and perhaps other experiment metadata
  def __init__(self, ID):
    self.ID = ID
    self.datatypes = {}

  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

class Eyetrack_Data:
  def __init__(self, path):
    self.path = path
    self.interpolation = None

    self.missing_frames = 0
    self.right_only_frames = 0
    self.left_only_frames = 0
    self.both_eyes_frames = 0

    with open(path, 'r') as file:
      # Allocate space for data, as (timestamp, x, y) for each timepoint
      print(path)
      self.raw_data = np.zeros((sum(1 for line in csv.reader(file, delimiter = ',')), 3))

    with open(path, 'r') as file:
      reader = csv.reader(file, delimiter = ',')
      for (frame_num, row) in enumerate(reader):

        # rows containing valid eyetracking data should have len(row) >= 7
        if len(row) < 7:
          continue

        # Eyetracking data before trial starts is only used for interpolating
        time, _, _, x_left, y_left, x_right, y_right = (np.asarray(row)).astype(np.float)
        self.raw_data[frame_num, 0] = time

        if x_left == 0:
          if x_right == 0:
            self.raw_data[frame_num, 1:3] = [float('nan'), float('nan')] # If neither eye was captured, record NaN
            self.missing_frames += 1
          else:
            self.raw_data[frame_num, 1:3] = [x_right, y_right] # Only right eye was captured
            self.right_only_frames += 1
        elif x_right == 0:
          self.raw_data[frame_num, 1:3] = [x_left, y_left] # Only left eye was captured
          self.left_only_frames += 1
        else:
          self.raw_data[frame_num, 1:3] = [(x_left + x_right)/2, (y_left + y_right)/2] # If both eyes were captured, use average
          self.both_eyes_frames += 1

    self.total_frames = self.missing_frames + self.right_only_frames + self.left_only_frames + self.both_eyes_frames
    self.proportion_total_missing_frames = float(self.missing_frames)/self.total_frames

  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)


class Eyetrack_Trial_Data:
  def __init__(self, data):
    self.data = data
    self.proportion_missing = np.mean(np.isnan(data[:, 1]))

  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)


# Stores all data about a single TrackIt experiment
class Trackit_Data:
  def __init__(self, path):
    self.path = path
    self.metadata = {}
    self.trials = []
    num_rows_after_BEGIN_New_Trial = -1
    with open(path, 'r') as file:
      reader = csv.reader(file, delimiter = ',')
      for (row_num, row) in enumerate(reader):
        # Code for reading in experiment metadata at top of TrackIt file
        if row_num == 0: # File metadata names row
          metadata_row = row
          continue
        elif row_num == 1: # File metadata values row
          for (metadata_name, metadata_value) in zip(metadata_row, row):
            self.metadata[metadata_name] = metadata_value
          continue

        # Done reading experiment metadata, start reading (meta)data from trials
        # Note that proper TrackIt data should never contain empty rows
        if row[0] == 'BEGIN New Trial':
          num_rows_after_BEGIN_New_Trial = 1
          self.trials.append(TrackIt_Trial_Data())

        elif row[0] == 'END New Trial':
          self.trials[-1].timestamps = np.asarray(self.trials[-1].timestamps)
          self.trials[-1].rel_timestamps = np.asarray(self.trials[-1].rel_timestamps)
          self.trials[-1].object_positions = np.asarray(self.trials[-1].object_positions)

        elif row[0] == 'End of entire file.':
          return # Indicate clean exit

        elif num_rows_after_BEGIN_New_Trial == 1: # Trial metadata names row
          num_rows_after_BEGIN_New_Trial += 1
          trial_metadata_row = row

        elif num_rows_after_BEGIN_New_Trial == 2: # Trial metadata values row
          num_rows_after_BEGIN_New_Trial += 1
          for (trial_metadata_name, trial_metadata_value) in zip(trial_metadata_row, row):
            self.trials[-1].trial_metadata[trial_metadata_name] = trial_metadata_value

        elif num_rows_after_BEGIN_New_Trial == 3: # Target object identifier row
          num_rows_after_BEGIN_New_Trial += 1
          self.trials[-1].target_index = (row.index('target')-2)/2
          self.trials[-1].num_objects = int((len(row) - 2)/2)
          for _ in range(self.trials[-1].num_objects):
            self.trials[-1].object_positions.append([])

        elif num_rows_after_BEGIN_New_Trial == 4: # Trial object names row
          num_rows_after_BEGIN_New_Trial += 1
          if row[-1] == 'Blinking object index':
            self.trials[-1].is_supervised = True
            self.trials[-1].meta_data['object_names'] = [name[:-2] for name in row[2:-1:2]]
            self.trials[-1].blinking_object_IDs = []
          else:
            self.trials[-1].is_supervised = False
            self.trials[-1].meta_data['object_names'] = [name[:-2] for name in row[2::2]]

        else: # Otherwise, the row should be a normal data row, and the first entry of the row should be the relative timestamp
          # test = int(row[0])
          self.trials[-1].rel_timestamps.append(int(row[0]))
          self.trials[-1].timestamps.append(int(row[1]))
          for object_index in range(self.trials[-1].num_objects):
            self.trials[-1].object_positions[object_index].append([float(row[2+(2*object_index)]), float(row[3+(2*object_index)])])
          if self.trials[-1].is_supervised:
            self.trials[-1].blinking_object_IDs.append(int(row[-1]))

  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

# Stores data about a single TrackIt trial
class TrackIt_Trial_Data:
  def __init__(self):
    self.meta_data = {}
    self.timestamps = []
    self.rel_timestamps = []
    self.trial_metadata = {}
    self.object_positions = []

  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

def load_dataset(experiment_ID, datatype_ID, subjects = {}):
  """Load a data set (a set of data files, as defined in dataset_list).
  
  If subjects is provided, data will be added to the data already contained in subjects
  (matching subject_IDs that already exist, and creating/appending new subjects for novel subject_IDs).

  experiment_ID -- (string) possible values are: 'shrinky' and 'noshrinky'
  datatype_ID -- (string) possible values are: 'trackit' and 'eyetrack'
  subjects -- (dict mapping subject IDs to Subject instances) dict to which to add new data
  
  """
  for subject_idx in range(50):
    subject_ID = str(subject_idx)
    path = '../' + datatype_ID + '/' + experiment_ID + '/' + subject_ID + '.csv'
    print('Loading ' + experiment_ID + '_' + datatype_ID + ' for subject ' + subject_ID)
    if not subject_ID in subjects:
      subjects[subject_ID] = Subject(subject_ID)
    if not experiment_ID in subjects[subject_ID].experiments:
      subjects[subject_ID].experiments[experiment_ID] = Experiment(experiment_ID)
    if datatype_ID in subjects[subject_ID].experiments[experiment_ID].datatypes:
      raise ValueError('Subject ' + subject_ID + ' already has data type ' + datatype_ID + \
                       ' for experiment ' + experiment_ID + '. Duplicate data types are not currently supported.')
    subjects[subject_ID].experiments[experiment_ID].datatypes[datatype_ID] = load_data(subject_ID, datatype_ID, path)
  return subjects

def load_data(subject_ID, datatype_ID, path):
  """Load a particular data file."""

  if datatype_ID == 'trackit':
    trackit_data = Trackit_Data(path)
    return trackit_data

  if datatype_ID == 'eyetrack':
    return Eyetrack_Data(path)

  raise ValueError('Unknown datatype ID: ' + datatype_ID)
