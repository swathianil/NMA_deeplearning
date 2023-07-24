import os
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import connectome

HCP_DIR = "./hcp_task"
rootpath='/content/gdrive/MyDrive/'
inputData=rootpath+'hcp_task/subjects/'
task='GAMBLING'

subjects=os.listdir(inputData)
N_SUBJECTS = 100

# The data have already been aggregated into ROIs from the Glasser parcellation
N_PARCELS = 360

# The acquisition parameters for all tasks were identical
TR = 0.72

EXPERIMENTS = {
    'MOTOR'      : {'cond':['lf','rf','lh','rh','t','cue']},
    'WM'         : {'cond':['0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools']},
    'EMOTION'    : {'cond':['fear','neut']},
    'GAMBLING'   : {'cond':['loss_event','win_event']},
    'LANGUAGE'   : {'cond':['math','story']},
    'RELATIONAL' : {'cond':['match','relation']},
    'SOCIAL'     : {'cond':['ment','rnd']}
}

HEMIS = ["Right", "Left"]

# Each experiment was repeated twice in each subject
RUNS   = ['LR','RL']
N_RUNS = 2


def get_normalization_stat():
  dataMat = []
  for i in range (len(subjects)):
      for j in range(len(RUNS)):
        ds = load_single_timeseries(subjects[i], task, j, remove_mean=False)
        if(i==0):
          dataMat=ds
        else:
          dataMat=np.hstack((dataMat,ds))

  region_mean=np.mean(dataMat,axis=1)
  region_std=np.std(dataMat,axis=1)
  mat_shape=dataMat.shape

  total_mat=np.reshape(dataMat,(1,mat_shape[0]*mat_shape[1]))
  global_mean=np.mean(total_mat)
  global_std=np.std(total_mat)

  return region_mean,region_std,global_mean,global_std



def load_single_timeseries(subject, experiment, run, remove_mean=True):
  """Load timeseries data for a single subject and single run.

  Args:
    subject (str):      subject ID to load
    experiment (str):   Name of experiment
    run (int):          (0 or 1)
    remove_mean (bool): If True, subtract the parcel-wise mean (typically the mean BOLD signal is not of interest)

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_run  = RUNS[run]
  bold_path = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{bold_run}"
  bold_file = "data.npy"
  ts = np.load(f"{bold_path}/{bold_file}")
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts


def load_evs(subject, experiment, run):
  """Load EVs (explanatory variables) data for one task experiment.

  Args:
    subject (str): subject ID to load
    experiment (str) : Name of experiment
    run (int): 0 or 1

  Returns
    evs (list of lists): A list of frames associated with each condition

  """
  frames_list = []
  task_key = f'tfMRI_{experiment}_{RUNS[run]}'
  for cond in EXPERIMENTS[experiment]['cond']:
    ev_file  = f"{HCP_DIR}/subjects/{subject}/{experiment}/{task_key}/EVs/{cond}.txt"
    ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
    ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
    # Determine when trial starts, rounded down
    start = np.floor(ev["onset"] / TR).astype(int)
    # Use trial duration to determine how many frames to include for trial
    duration = np.ceil(ev["duration"] / TR).astype(int)
    # Take the range of frames that correspond to this specific trial
    frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
    frames_list.append(frames)

  return frames_list

## loading timeseries per trial
def load_timeserise(task, subjectID, run, normalization=False, normalStat=[], normalType='global'):
    right_Hemi_idx=np.arange(0,int(N_PARCELS/2),1)
    left_Hemi_idx=np.arange(int(N_PARCELS/2),N_PARCELS,1)
    data = load_single_timeseries(subject=subjectID, experiment=task, run=run, remove_mean=False)
    evs = load_evs(subject=subjectID, experiment=task, run=run)

    if(normalization):
      if(normalType=='global'): 
        data=(data-normalStat['mean'])/normalStat['std']
      
      else:
        for i in range(N_PARCELS):
          data[i,:]=(data[i,:]-normalStat['mean'][i])/normalStat['std'][i]

    ## separating retions within left and right hemisphere
    timeserise_right=data[right_Hemi_idx,:]
    timeserise_left=data[left_Hemi_idx,:]
    regionNum=len(right_Hemi_idx)
    Total_Timeseries_Left=[]
    Total_Timeseries_Right=[]
    Cond_Label=[]
    for cond in len(EXPERIMENTS[task]['cond']):
      trialIndex=evs[cond]
      trlLen=len(trialIndex)
      sampleNo=np.shape(trialIndex[0])[0]

      timeserise_right_trials=np.array((regionNum, trlLen, sampleNo))
      timeserise_left_trials=np.array((regionNum, trlLen, sampleNo))
      for t in range(trlLen):
        point_index=trialIndex[t]
        timeserise_right_trials[:,t,:]=timeserise_right[:,point_index]
        timeserise_left_trials[:,t,:]=timeserise_left[:,point_index]

      Cond_Label.append(EXPERIMENTS[task]['cond'])
      if(cond==0):
        Total_Timeseries_Left=timeserise_left_trials
        Total_Timeseries_Right=timeserise_right_trials
      else:
        Total_Timeseries_Left=np.hstack((Total_Timeseries_Left, timeserise_left_trials))
        Total_Timeseries_Right=np.hstack((Total_Timeseries_Right, timeserise_right_trials))

    return Total_Timeseries_Left, Total_Timeseries_Right


def my_subject_connectivity(timeseries, kind):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['TPE', 'TE', 'correlation','partial correlation']:
        if kind not in ['TPE', 'TE']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)


        if kind not in ['TPE', 'TE']:
            return connectivity
        else:
            return connectivity_fit
        
def trials_connectivity(timeseries):
  regNo,trlNo,smpl=timeseries.shape

  network=[]
  for t in range(trlNo):
     trial_data=timeseries[:,t,:]
     trial_data=trial_data.transpose()
     corr = my_subject_connectivity(timeseries = [trial_data], kind = 'correlation')
     network.append(corr[0])

  return network
  

