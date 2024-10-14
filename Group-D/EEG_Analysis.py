import scipy.io
import os
import scipy.io
import mne
import numpy as np
from mne.conftest import event_id
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
import warnings
import matplotlib
matplotlib.use('TkAgg')
from numpy.ma.extras import average
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# %matplotlib inline

# %%
############################
#### Step 1: Input file ####
############################

# Get File Directory (Change to your own directory)
eegname = "C_GroupD_Users_Coen_Documents_BDFdata_Testdata.bdf"
eegpath = "/Users/jiahuan/Desktop/NYU/PSYCH-GA 3405 EEG/Datasets/GroupD/EEG/Recording/"
raw = mne.io.read_raw_bdf(eegpath+eegname, preload=True)

# %%
# Basic information (4584 second, 272 channels)
print(raw.info)
# print raw time series data (0-5s, first 10 channels)
raw.plot(duration=5, n_channels=10, clipping=None)


#########################
#### Step 2: Montage ####
#########################
# %%
# We are using BioSemi 256 electrode setting (A1-A32...H1-H32)
# EXG1-8 for EOG and reference
# GSR1’, ‘GSR2’ ‘Erg1’, ‘Erg2’ ‘Resp’ ‘Plet’ ‘Temp’ not used
# 'Status' is machine light

print(f"Number of electrodes: {len(raw.ch_names)}") # 256 + 8 + 7 + 1 = 272

# Set non-EEG Channels
non_eeg_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
raw.set_channel_types({
    'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog',  # 眼动电图
    'EXG5': 'misc', 'EXG6': 'misc', 'EXG7': 'misc', 'EXG8': 'misc',  # 参考或其他
    'GSR1': 'misc', 'GSR2': 'misc',
    'Erg1': 'misc', 'Erg2': 'misc',
    'Resp': 'resp', 'Plet': 'misc', 'Temp': 'misc'
})

# Load MNE default setting for BioSemi 256
montage = mne.channels.make_standard_montage('biosemi256')
raw.set_montage(montage)  # montage!
print(raw.info)  # Check new channels
# chs: 256 EEG, 4 EOG, 10 misc, 1 Respiration monitoring channel, 1 Stimulus

# Draw spatial lead plot
raw.plot_sensors(ch_type='eeg', show_names=True)

# %%
###########################
#### Step 3: Filtering ####
###########################

# Check environmental noise
# We don't need to remove 60Hz power frequency interference, because the room is well isolated!
raw.plot_psd(average=True)

# %%
# Highpass and lowpass filtering, keep frequency betwee 0.1 ~ 30 Hz (normal EEG range)
# Default method is FIR
# Use method = iir, to use IIR filter
raw = raw.filter(l_freq=0.1, h_freq=30)


# %%
# Draw power plot again
# If "average = False" will print all channels independently
raw.plot_psd(average=True)

# %%
#################################
#### Step 4: Remove artifact ####
#################################
####################################################
### Don't do this part, need furthur discuss!!! ###
###################################################
# Open an interactive window, manually mark the bad epoch
fig = raw.plot()
fig.fake_kepress('a')

# %%
# Mark bad channel
# raw.info['bads'].append('FC5')
# print(raw.info['bads'])
# raw.info['bads'].append('C3')
# print(raw.info['bads'])

# %%
# Interpolate bad channels
raw = raw.interpolate_bads(reset_bads=True)

# %%
# use ICA to exclude EOG artifact
# First apply on 1 ~ 30 Hz data
# Second apply on 0.1 ~ 1 Hz data, because bad at combined
ica = ICA(max_iter='auto')
raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
ica.fit(raw_for_ica)

# %%
# Plot ICA result, each row is an ICA component
ica.plot_sources(raw_for_ica)

# %%
# Plot ICA brain map, each brain is an ICA component
ica.plot_components()

# %%
# Remove ICA 001, and see how is the raw data changed
ica.plot_overlay(raw_for_ica, exclude=[1])

# %%
# Observe the ICA 001 alone
ica.plot_properties(raw, picks=[1])

# %%
# Remove ICA 001, which is a EOG component, because large power at low frequency,
# located at frontal lobe, enhanced in some trails

ica.exclude = [1]
ica.apply(raw)
print(raw.info)

# %%
# Draw plot after ICA
raw.plot(duration=5, n_channels=32, clipping=None)

# %%
#####################################################
#### Step 5: Rereferencing (takes 30 minutes!!!) ####
#####################################################
#
# # Use 'EXG1', 'EXG2', 'EXG3', 'EXG4' electrode to rereference
# raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2', 'EXG3', 'EXG4'])

# Use average reference (recommend for large dataset with more than 128 channels, but takes 30 minutes!!!)
raw.set_eeg_reference(ref_channels='average')

# # Use bipolar reference
# raw_bip_ref = mne.set_bipolar_reference(raw, anode=['EEG X'], cathode=['EEG Y'])

# %%
##############################
#### Step 6: Data Segment ####
##############################

def load_markers_from_mat(mat_file_path):
    """Dynamically extract marker data from a .mat file, including event name, onsetTime, and offsetTime"""
    mat_data = scipy.io.loadmat(mat_file_path)

    # Dynamically search for a key containing 'timingData', supporting keys like timingData3 or similar
    timing_data_key = [key for key in mat_data.keys() if 'timingData' in key][0]
    timing_data = mat_data[timing_data_key]

    events = []

    # Iterate through each event, dynamically extract data based on field names
    for event in timing_data[0]:
        # Try extracting event name from 'stiType' or 'event'
        if 'stiType' in timing_data.dtype.names:
            event_name = event[timing_data.dtype.names.index('stiType')][0]
        elif 'event' in timing_data.dtype.names:
            event_name = event[timing_data.dtype.names.index('event')][0]
        else:
            raise ValueError("No 'stiType' or 'event' field found")

        # Extract onsetTime
        onset_index = timing_data.dtype.names.index('onsetTime')
        onset_time = event[onset_index][0][0]

        # Extract offsetTime (if available), otherwise set duration to 0
        if 'offsetTime' in timing_data.dtype.names:
            offset_index = timing_data.dtype.names.index('offsetTime')
            offset_time = event[offset_index][0][0]
        else:
            offset_time = onset_time  # If no offsetTime, set duration to 0

        # Add the extracted information to the events list
        events.append((event_name, onset_time, offset_time))

    return events


def create_annotations_from_events(events):
    """Create annotations from marker data"""
    onset_times = [event[1] for event in events]  # Extract onset times
    durations = [max(event[2] - event[1], 0) for event in events]  # Ensure non-negative durations
    descriptions = [event[0] for event in events]  # Extract event names as descriptions

    # Create MNE annotations
    annotations = mne.Annotations(onset=onset_times, duration=durations, description=descriptions)

    return annotations


def load_all_mat_files_and_create_annotations(directory_path):
    """Load all .mat files in a folder, combine markers into annotations"""
    all_events = []

    # Iterate over all .mat files in the folder
    for filename in os.listdir(directory_path):
        if filename.endswith(".mat"):
            file_path = os.path.join(directory_path, filename)

            # Load markers from the file
            events = load_markers_from_mat(file_path)

            # Add the events from the current file to the total events list
            all_events.extend(events)

    # Create annotations
    annotations = create_annotations_from_events(all_events)

    return annotations


#%%
# Change to your own marker directory
directory_path = marker_folder = "/Users/jiahuan/Desktop/NYU/PSYCH-GA 3405 EEG/Datasets/GroupD/EEG/TimeStamps/"
#%%
# Load markers from all .mat files and create annotations
annotations = load_all_mat_files_and_create_annotations(directory_path)

#%%
# Add the combined annotations to the raw data
raw.set_annotations(annotations)
#%%
# Check the added annotations
print(raw.annotations)

# Visualize the combined data (with annotations)
raw.plot()






# %%
# Convert Annotation information into Event (put it outside from Raw)
events, event_id = mne.events_from_annotations(raw)

# %%
# Make segments/ epochs
# -1 ~ 2s is the window of an event, event is "square"
# baseline takes values from 0.5s before an event
# If amplitude within one epoch is larger than 2e-4, that trial will be exclude
epochs = mne.Epochs(raw, events, event_id=2, tmin=-1, tmax=2, baseline=(-0.5, 0),
                    preload=True, reject=dict(eeg=2e-4))

print(epochs)

#%%
# Visualize ERP/ event evoked potential, show 4 ERPs here
epochs.plot(n_epochs=4)

#%%
# Visualize Power spectrum density (PSD plot) (during one epoch)
epochs.compute_psd().plot(picks = 'eeg')

#%%
# Visualize Power spectrum (based on band alpha, beta, theta)
bands = [(4,8,'Theta'),(8,12,'Alpha'),(12,30,'Beta')]
epochs.plot_psd_topomap(bands=bands, vlim = 'joint')

#%%
#####################################
### Step 7: Superimposed average ####
#####################################

# Get average ERP
evoked = epochs.average()

# Visualize average ERP
evoked.plot()

#%%
# Visualize ERP topomap, at timepoint 0, 2, 5 of event
times = np.linspace(0,2,5)
evoked.plot_topomap(times = times, colorbar=True)

#%%
# Visualize ERP topomap, at t=0.75-0.85
evoked.plot_topomap(times = 0.8, average = 0.1)

#%%
# Combine topomap and Power spectrum
evoked.plot_joint()

#%%
# Heatmap for each channel, during Event
evoked.plot_image()

#%%
# Topo plot during ERP
evoked.plot_topo()

#%%
# Average all electrode, draw a ERP power spectrum
mne.viz.plot_compare_evokeds(evokeds = evoked, combine='mean')

#%%
# Average ERP power spectrum for occipital lobe
mne.viz.plot_compare_evokeds(evokeds = evoked, picks=['O1','O2','Oz'],combine='mean')


#%%
#########################################
#### Step 8:Time Spectrum Analysis ######
#########################################



#%%
################################
#### Step 9: Extract Data ######
################################


