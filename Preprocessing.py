# -*- coding: utf-8 -*-
"""
Created on Jan 2022

@author: Francisco Viana
"""

# %% Define Paths

code_path = 'C:/Users/Admin/Desktop/TMS_preprocessing_pipeline/'
# data_path = 'C:/Users/Admin/Desktop/TMS_preprocessing_pipeline/data/'
data_path = 'D:/FV/Tese/Code/raw_EMGdata/'

# %% Import Packages
import os
import sys
sys.path.insert(0,code_path)
from TMS_preprocessing_module import *
import numpy as np
from math import nan

# %% Define file for preprocessing

# CHANGE FILE NAME BELOW!
name = 'HC41_s3_l_2021-10-01_18-50-16'
# =========================================

rawData_path = data_path + name + ".bin"
output_path = data_path + 'preprocessed/'

if not os.path.exists(output_path):
    # Create a new directory for preprocessed data if it does not exist 
    os.makedirs(output_path)
    os.makedirs(output_path + 'XLSX/')
    os.makedirs(output_path + 'NPY/')


info = get_info(rawData_path)   # Get session information
FS = 4000   # Sampling frequency (Hz)
data = DenoiseSignal(load_data(rawData_path,fs = FS))
# Print session information
print('-'*20)
print('Preprocessing file:')
print('\tSubject: {};\n\tSession: {};\n\tCollected side: {};\n\tCollection Date: {}.'.format(info[0],info[1],info[2],info[3]))
print('-'*20)
view_channels(data,FS)

# %% Split Trial
'''
  For each variable (l_rmt, mvic, etc.), type the time (in minutes) corresponding 
to the end of the section in question.
  In the case of "ref_emg", type the time where you want to extract a 500 ms 
emg trace for baseline noise reference.

Use at least 2 decimal points
'''
# CHANGE BELOW!
plt.close('all')
ref_emg = 55.1
l_rmt = 6.94
mvic = 10.23
l_amt = 17.22
r_rmt = nan
b_meps = 22.5
itbs = 27.5
t0_meps = 32.86
rest0 = 37.25
t10_meps = 42.61
rest10 = 47.29
t20_meps = 52.77
rest20 = 57.28
t30_meps = 62.32
r_rmt_pos = nan
l_rmt_pos = 64.96
# =========================================

markers = {'ref_emg':ref_emg, 'l_rmt':l_rmt ,'mvic':mvic ,'l_amt':l_amt ,
           'r_rmt':r_rmt ,'b_meps':b_meps ,'itbs':itbs,'t0_meps':t0_meps ,
           'rest0':rest0 ,'t10_meps':t10_meps ,'rest10':rest10 ,
           't20_meps':t20_meps ,'rest20':rest20 ,'t30_meps':t30_meps ,
           'r_rmt_pos':r_rmt_pos ,'l_rmt_pos':l_rmt_pos}

#split session and plot MVICs
DATA, mvicData = prepDataDiccionary(data,markers,FS)


# %% Segment MVICs

plt.close('all')

'''
Type the time (in seconds) corresponding to the begining of each MVIC
'''
# CHANGE BELOW!
mvic_start_1 = 30.5
mvic_start_2 = 92.2
mvic_start_3 = 152.50
# =========================================

CONTRACTIONS = plot_MVICs(mvicData,[mvic_start_1,mvic_start_2,mvic_start_3])

# %% Check for muscle activity and plot overlapped MEPs

PULSES, FLAGS = check_muscle_activity(DATA,info,fs = FS)

plot_MEPoverlap(DATA,PULSES,info,fs = FS)

# %% Plot baseline MEPs
plt.close('all')

''' 
Type the time (in miliseconds) corresponding to an interval where all the MEPs
are contained.
'''
# CHANGE BELOW! (if needed)
mep_start = 20.27
mep_end = 32.2


EXTREMES = {}

# ================================== b_meps =================================
key = 'b_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

# %% Check b_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')
# %% Plot T0 MEPs

# ================================== t0_meps =================================
key = 't0_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t0_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Plot T10 MEPs

# ================================== t10_meps =================================
key = 't10_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t10_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Plot T20 MEPs

# ================================== t20_meps =================================
key = 't20_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t20_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Plot T30 MEPs

# ================================== t30_meps =================================
key = 't30_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t30_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Create Peak Excel

# Create excell with MEP peak values
xlsx_path = create_MEPeacks_excel(EXTREMES,info,output_path)

# Plot MEPs that where mischarachterized by algorithm
plot_badMEPs(mep_start,mep_end,info,DATA,PULSES,EXTREMES,FLAGS,fs = FS)


'''
If some MEPs where badly identified, follow these steps:
    1. Open corresponding .xlsx file (ex: HC36_s1_l_peaks.xlsx)
    2. Check the correct value in the plotted MEPs
    3. Make changes in excel file and save document and close.
    4. Run next section
'''
# %% Correct, Save and export results

plt.close('all')

SavePreprocessedData(xlsx_path, output_path, DATA, PULSES, EXTREMES, 
                     CONTRACTIONS, FS, FLAGS, info, markers)
