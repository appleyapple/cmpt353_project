import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal.filter_design import butter
import statsmodels.api as sm
from scipy import signal

# to remove a figure producing limit warning
mpl.rcParams['figure.max_open_warning'] = 0

# function to load data from csv file
def load_data(csv):
    data = pd.read_csv('data/'+csv, header=0, names=['time', 'x', 'y', 'z', 't'])
    data = data.drop(columns=['t'])
    return data


# load data from csv
print('Loading datasets...')
left_standard = load_data('left_standard.csv')
right_standard = load_data('right_standard.csv')
left_drag = load_data('left_drag.csv')
right_drag = load_data('right_drag.csv')
left_limp = load_data('left_limp.csv')
right_limp = load_data('right_limp.csv')
left_waddle = load_data('left_waddle.csv')
right_waddle = load_data('right_waddle.csv')

datasets = [left_standard, right_standard, left_drag, right_drag, left_limp, right_limp, left_waddle, right_waddle]
datasets_str = ['left_standard', 'right_standard', 'left_drag', 'right_drag', 'left_limp', 'right_limp', 'left_waddle', 'right_waddle']


# output folders init
print('Initializing output folders...')
output_folders = ['loess_smooth', 'butterworth_filter']
for folder in output_folders:
    try:
        os.makedirs('output/' + folder)
    except:
        print('Output folder ' + folder + ' already exists.')


# loess smoothing
print('LOESS smoothing...')
loess_smoothed_datasets_x = []
loess_smoothed_datasets_y = []
loess_smoothed_datasets_z = []
lowess = sm.nonparametric.lowess

#for x-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = lowess(dataset['x'], dataset['time'], frac=0.025)
    loess_smoothed_datasets_x.append(smoothed_dataset)
    plt.figure(figsize=(12, 8))
    plt.title(name + ' x-axis acceleration over time')
    plt.plot(dataset['time'], dataset['x'], alpha=0.5)
    plt.plot(dataset['time'], smoothed_dataset[:,1], alpha=0.8)
    plt.legend(['Raw Data', 'LOESS-Smoothed Data'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('x (m/s^2)')
    plt.savefig('output/loess_smooth/' + name + '.png')

#for y-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = lowess(dataset['y'], dataset['time'], frac=0.025)
    loess_smoothed_datasets_y.append(smoothed_dataset)
    plt.figure(figsize=(12, 8))
    plt.title(name + ' y-axis acceleration over time')
    plt.plot(dataset['time'], dataset['y'], alpha=0.5)
    plt.plot(dataset['time'], smoothed_dataset[:,1], alpha=0.8)
    plt.legend(['Raw Data', 'LOESS-Smoothed Data'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('y (m/s^2)')
    plt.savefig('output/loess_smooth/' + name + '_y.png')

#for z-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = lowess(dataset['z'], dataset['time'], frac=0.025)
    loess_smoothed_datasets_z.append(smoothed_dataset)
    plt.figure(figsize=(12, 8))
    plt.title(name + ' z-axis acceleration over time')
    plt.plot(dataset['time'], dataset['z'], alpha=0.5)
    plt.plot(dataset['time'], smoothed_dataset[:,1], alpha=0.8)
    plt.legend(['Raw Data', 'LOESS-Smoothed Data'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('z (m/s^2)')
    plt.savefig('output/loess_smooth/' + name + '_z.png')

#loess dataframes ls = left standard, ld = left drag, ll = left limp, lw = left waddle, etc..
loess_ls = pd.DataFrame({'time':loess_smoothed_datasets_x[0][:,0], 'x':loess_smoothed_datasets_x[0][:,1], 'y':loess_smoothed_datasets_y[0][:,1],'z':loess_smoothed_datasets_z[0][:,1]})
loess_rs = pd.DataFrame({'time':loess_smoothed_datasets_x[1][:,0],'x':loess_smoothed_datasets_x[1][:,1], 'y':loess_smoothed_datasets_y[1][:,1],'z':loess_smoothed_datasets_z[1][:,1]}) 
loess_ld = pd.DataFrame({'time':loess_smoothed_datasets_x[2][:,0],'x':loess_smoothed_datasets_x[2][:,1], 'y':loess_smoothed_datasets_y[2][:,1],'z':loess_smoothed_datasets_z[2][:,1]})
loess_rd = pd.DataFrame({'time':loess_smoothed_datasets_x[3][:,0],'x':loess_smoothed_datasets_x[3][:,1], 'y':loess_smoothed_datasets_y[3][:,1],'z':loess_smoothed_datasets_z[3][:,1]})
loess_ll = pd.DataFrame({'time':loess_smoothed_datasets_x[4][:,0],'x':loess_smoothed_datasets_x[4][:,1], 'y':loess_smoothed_datasets_y[4][:,1],'z':loess_smoothed_datasets_z[4][:,1]})
loess_rl = pd.DataFrame({'time':loess_smoothed_datasets_x[5][:,0],'x':loess_smoothed_datasets_x[5][:,1], 'y':loess_smoothed_datasets_y[5][:,1],'z':loess_smoothed_datasets_z[5][:,1]}) 
loess_lw = pd.DataFrame({'time':loess_smoothed_datasets_x[6][:,0],'x':loess_smoothed_datasets_x[6][:,1], 'y':loess_smoothed_datasets_y[6][:,1],'z':loess_smoothed_datasets_z[6][:,1]}) 
loess_rw = pd.DataFrame({'time':loess_smoothed_datasets_x[7][:,0],'x':loess_smoothed_datasets_x[7][:,1], 'y':loess_smoothed_datasets_y[7][:,1],'z':loess_smoothed_datasets_z[7][:,1]})


# butterworth filter
print('Butterworth filtering...')
butter_smoothed_datasets_x = []
butter_smoothed_datasets_y = []
butter_smoothed_datasets_z = []
b,a = signal.butter(5, 0.012, btype='lowpass', analog=False)

#for x-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = signal.filtfilt(b, a, dataset['x'])
    butter_smoothed_datasets_x.append(smoothed_dataset)
    plt.figure(figsize=(12, 8))
    plt.title(name + ' x-axis acceleration over time')
    plt.plot(dataset['time'], dataset['x'], alpha=0.5)
    plt.plot(dataset['time'], smoothed_dataset[:], alpha=0.8)
    plt.legend(['Raw Data', 'Butterworth Filtered Data'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('x (m/s^2)')
    plt.savefig('output/butterworth_filter/' + name + '.png')

#for y-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = signal.filtfilt(b, a, dataset['y'])
    butter_smoothed_datasets_y.append(smoothed_dataset)
    plt.figure(figsize=(12, 8))
    plt.title(name + ' y-axis acceleration over time')
    plt.plot(dataset['time'], dataset['y'], alpha=0.5)
    plt.plot(dataset['time'], smoothed_dataset[:], alpha=0.8)
    plt.legend(['Raw Data', 'Butterworth Filtered Data'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('y (m/s^2)')
    plt.savefig('output/butterworth_filter/' + name + '_y.png')

#for z-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = signal.filtfilt(b, a, dataset['z'])
    butter_smoothed_datasets_z.append(smoothed_dataset)
    plt.figure(figsize=(12, 8))
    plt.title(name + ' z-axis acceleration over time')
    plt.plot(dataset['time'], dataset['z'], alpha=0.5)
    plt.plot(dataset['time'], smoothed_dataset[:], alpha=0.8)
    plt.legend(['Raw Data', 'Butterworth Filtered Data'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('z (m/s^2)')
    plt.savefig('output/butterworth_filter/' + name + '_z.png')

#butterworth dataframes ls = left standard, ld = left drag, ll = left limp, lw = left waddle, etc..
butter_ls = pd.DataFrame({'time':loess_smoothed_datasets_x[0][:,0],'x':butter_smoothed_datasets_x[0], 'y':butter_smoothed_datasets_y[0],'z':butter_smoothed_datasets_z[0]})
butter_rs = pd.DataFrame({'time':loess_smoothed_datasets_x[1][:,0],'x':butter_smoothed_datasets_x[1]*(-1), 'y':butter_smoothed_datasets_y[1]*(-1),'z':butter_smoothed_datasets_z[1]*(-1)}) 
butter_ld = pd.DataFrame({'time':loess_smoothed_datasets_x[2][:,0],'x':butter_smoothed_datasets_x[2], 'y':butter_smoothed_datasets_y[2],'z':butter_smoothed_datasets_z[2]})
butter_rd = pd.DataFrame({'time':loess_smoothed_datasets_x[3][:,0],'x':butter_smoothed_datasets_x[3]*(-1), 'y':butter_smoothed_datasets_y[3]*(-1),'z':butter_smoothed_datasets_z[3]*(-1)})
butter_ll = pd.DataFrame({'time':loess_smoothed_datasets_x[4][:,0],'x':butter_smoothed_datasets_x[4], 'y':butter_smoothed_datasets_y[4],'z':butter_smoothed_datasets_z[4]})
butter_rl = pd.DataFrame({'time':loess_smoothed_datasets_x[5][:,0],'x':butter_smoothed_datasets_x[5]*(-1), 'y':butter_smoothed_datasets_y[5]*(-1),'z':butter_smoothed_datasets_z[5]*(-1)}) 
butter_lw = pd.DataFrame({'time':loess_smoothed_datasets_x[6][:,0],'x':butter_smoothed_datasets_x[6], 'y':butter_smoothed_datasets_y[6],'z':butter_smoothed_datasets_z[6]}) 
butter_rw = pd.DataFrame({'time':loess_smoothed_datasets_x[7][:,0],'x':butter_smoothed_datasets_x[7]*(-1), 'y':butter_smoothed_datasets_y[7]*(-1),'z':butter_smoothed_datasets_z[7]*(-1)})

# calculate x velocity
butter_ls['x_delta_velocity'] = butter_ls['x'] * (butter_ls['time'] - butter_ls['time'].shift(1))
butter_ls['x_delta_velocity'][0] = 0
butter_ls['x_velocity'] = butter_ls['x_delta_velocity'].cumsum()

butter_rs['x_delta_velocity'] = butter_rs['x'] * (butter_rs['time'] - butter_rs['time'].shift(1))
butter_rs['x_delta_velocity'][0] = 0
butter_rs['x_velocity'] = butter_rs['x_delta_velocity'].cumsum()

butter_ld['x_delta_velocity'] = butter_ld['x'] * (butter_ld['time'] - butter_ld['time'].shift(1))
butter_ld['x_delta_velocity'][0] = 0
butter_ld['x_velocity'] = butter_ld['x_delta_velocity'].cumsum()

butter_rd['x_delta_velocity'] = butter_rd['x'] * (butter_rd['time'] - butter_rd['time'].shift(1))
butter_rd['x_delta_velocity'][0] = 0
butter_rd['x_velocity'] = butter_rd['x_delta_velocity'].cumsum()

butter_ll['x_delta_velocity'] = butter_ll['x'] * (butter_ll['time'] - butter_ll['time'].shift(1))
butter_ll['x_delta_velocity'][0] = 0
butter_ll['x_velocity'] = butter_ll['x_delta_velocity'].cumsum()

butter_rl['x_delta_velocity'] = butter_rl['x'] * (butter_rl['time'] - butter_rl['time'].shift(1))
butter_rl['x_delta_velocity'][0] = 0
butter_rl['x_velocity'] = butter_rl['x_delta_velocity'].cumsum()

butter_lw['x_delta_velocity'] = butter_lw['x'] * (butter_lw['time'] - butter_lw['time'].shift(1))
butter_lw['x_delta_velocity'][0] = 0
butter_lw['x_velocity'] = butter_lw['x_delta_velocity'].cumsum()

butter_rw['x_delta_velocity'] = butter_rw['x'] * (butter_rw['time'] - butter_rw['time'].shift(1))
butter_rw['x_delta_velocity'][0] = 0
butter_rw['x_velocity'] = butter_rw['x_delta_velocity'].cumsum()

# calculate y velocity
butter_ls['y_delta_velocity'] = butter_ls['y'] * (butter_ls['time'] - butter_ls['time'].shift(1))
butter_ls['y_delta_velocity'][0] = 0
butter_ls['y_velocity'] = butter_ls['y_delta_velocity'].cumsum()

butter_rs['y_delta_velocity'] = butter_rs['y'] * (butter_rs['time'] - butter_rs['time'].shift(1))
butter_rs['y_delta_velocity'][0] = 0
butter_rs['y_velocity'] = butter_rs['y_delta_velocity'].cumsum()

butter_ld['y_delta_velocity'] = butter_ld['y'] * (butter_ld['time'] - butter_ld['time'].shift(1))
butter_ld['y_delta_velocity'][0] = 0
butter_ld['y_velocity'] = butter_ld['y_delta_velocity'].cumsum()

butter_rd['y_delta_velocity'] = butter_rd['y'] * (butter_rd['time'] - butter_rd['time'].shift(1))
butter_rd['y_delta_velocity'][0] = 0
butter_rd['y_velocity'] = butter_rd['y_delta_velocity'].cumsum()

butter_ll['y_delta_velocity'] = butter_ll['y'] * (butter_ll['time'] - butter_ll['time'].shift(1))
butter_ll['y_delta_velocity'][0] = 0
butter_ll['y_velocity'] = butter_ll['y_delta_velocity'].cumsum()

butter_rl['y_delta_velocity'] = butter_rl['y'] * (butter_rl['time'] - butter_rl['time'].shift(1))
butter_rl['y_delta_velocity'][0] = 0
butter_rl['y_velocity'] = butter_rl['y_delta_velocity'].cumsum()

butter_lw['y_delta_velocity'] = butter_lw['y'] * (butter_lw['time'] - butter_lw['time'].shift(1))
butter_lw['y_delta_velocity'][0] = 0
butter_lw['y_velocity'] = butter_lw['y_delta_velocity'].cumsum()

butter_rw['y_delta_velocity'] = butter_rw['y'] * (butter_rw['time'] - butter_rw['time'].shift(1))
butter_rw['y_delta_velocity'][0] = 0
butter_rw['y_velocity'] = butter_rw['y_delta_velocity'].cumsum()

# calculate z velocity
butter_ls['z_delta_velocity'] = butter_ls['z'] * (butter_ls['time'] - butter_ls['time'].shift(1))
butter_ls['z_delta_velocity'][0] = 0
butter_ls['z_velocity'] = butter_ls['z_delta_velocity'].cumsum()

butter_rs['z_delta_velocity'] = butter_rs['z'] * (butter_rs['time'] - butter_rs['time'].shift(1))
butter_rs['z_delta_velocity'][0] = 0
butter_rs['z_velocity'] = butter_rs['z_delta_velocity'].cumsum()

butter_ld['z_delta_velocity'] = butter_ld['z'] * (butter_ld['time'] - butter_ld['time'].shift(1))
butter_ld['z_delta_velocity'][0] = 0
butter_ld['z_velocity'] = butter_ld['z_delta_velocity'].cumsum()

butter_rd['z_delta_velocity'] = butter_rd['z'] * (butter_rd['time'] - butter_rd['time'].shift(1))
butter_rd['z_delta_velocity'][0] = 0
butter_rd['z_velocity'] = butter_rd['z_delta_velocity'].cumsum()

butter_ll['z_delta_velocity'] = butter_ll['z'] * (butter_ll['time'] - butter_ll['time'].shift(1))
butter_ll['z_delta_velocity'][0] = 0
butter_ll['z_velocity'] = butter_ll['z_delta_velocity'].cumsum()

butter_rl['z_delta_velocity'] = butter_rl['z'] * (butter_rl['time'] - butter_rl['time'].shift(1))
butter_rl['z_delta_velocity'][0] = 0
butter_rl['z_velocity'] = butter_rl['z_delta_velocity'].cumsum()

butter_lw['z_delta_velocity'] = butter_lw['z'] * (butter_lw['time'] - butter_lw['time'].shift(1))
butter_lw['z_delta_velocity'][0] = 0
butter_lw['z_velocity'] = butter_lw['z_delta_velocity'].cumsum()

butter_rw['z_delta_velocity'] = butter_rw['z'] * (butter_rw['time'] - butter_rw['time'].shift(1))
butter_rw['z_delta_velocity'][0] = 0
butter_rw['z_velocity'] = butter_rw['z_delta_velocity'].cumsum()

# calculate x position
butter_ls['x_delta_position'] = butter_ls['x_velocity'] * (butter_ls['time'] - butter_ls['time'].shift(1))
butter_ls['x_delta_position'][0] = 0
butter_ls['x_position'] = butter_ls['x_delta_position'].cumsum()

butter_rs['x_delta_position'] = butter_rs['x_velocity'] * (butter_rs['time'] - butter_rs['time'].shift(1))
butter_rs['x_delta_position'][0] = 0
butter_rs['x_position'] = butter_rs['x_delta_position'].cumsum()

butter_ld['x_delta_position'] = butter_ld['x_velocity'] * (butter_ld['time'] - butter_ld['time'].shift(1))
butter_ld['x_delta_position'][0] = 0
butter_ld['x_position'] = butter_ld['x_delta_position'].cumsum()

butter_rd['x_delta_position'] = butter_rd['x_velocity'] * (butter_rd['time'] - butter_rd['time'].shift(1))
butter_rd['x_delta_position'][0] = 0
butter_rd['x_position'] = butter_rd['x_delta_position'].cumsum()

butter_ll['x_delta_position'] = butter_ll['x_velocity'] * (butter_ll['time'] - butter_ll['time'].shift(1))
butter_ll['x_delta_position'][0] = 0
butter_ll['x_position'] = butter_ll['x_delta_position'].cumsum()

butter_rl['x_delta_position'] = butter_rl['x_velocity'] * (butter_rl['time'] - butter_rl['time'].shift(1))
butter_rl['x_delta_position'][0] = 0
butter_rl['x_position'] = butter_rl['x_delta_position'].cumsum()

butter_lw['x_delta_position'] = butter_lw['x_velocity'] * (butter_lw['time'] - butter_lw['time'].shift(1))
butter_lw['x_delta_position'][0] = 0
butter_lw['x_position'] = butter_lw['x_delta_position'].cumsum()

butter_rw['x_delta_position'] = butter_rw['x_velocity'] * (butter_rw['time'] - butter_rw['time'].shift(1))
butter_rw['x_delta_position'][0] = 0
butter_rw['x_position'] = butter_rw['x_delta_position'].cumsum()

# calculate y position
butter_ls['y_delta_position'] = butter_ls['y_velocity'] * (butter_ls['time'] - butter_ls['time'].shift(1))
butter_ls['y_delta_position'][0] = 0
butter_ls['y_position'] = butter_ls['y_delta_position'].cumsum()

butter_rs['y_delta_position'] = butter_rs['y_velocity'] * (butter_rs['time'] - butter_rs['time'].shift(1))
butter_rs['y_delta_position'][0] = 0
butter_rs['y_position'] = butter_rs['y_delta_position'].cumsum()

butter_ld['y_delta_position'] = butter_ld['y_velocity'] * (butter_ld['time'] - butter_ld['time'].shift(1))
butter_ld['y_delta_position'][0] = 0
butter_ld['y_position'] = butter_ld['y_delta_position'].cumsum()

butter_rd['y_delta_position'] = butter_rd['y_velocity'] * (butter_rd['time'] - butter_rd['time'].shift(1))
butter_rd['y_delta_position'][0] = 0
butter_rd['y_position'] = butter_rd['y_delta_position'].cumsum()

butter_ll['y_delta_position'] = butter_ll['y_velocity'] * (butter_ll['time'] - butter_ll['time'].shift(1))
butter_ll['y_delta_position'][0] = 0
butter_ll['y_position'] = butter_ll['y_delta_position'].cumsum()

butter_rl['y_delta_position'] = butter_rl['y_velocity'] * (butter_rl['time'] - butter_rl['time'].shift(1))
butter_rl['y_delta_position'][0] = 0
butter_rl['y_position'] = butter_rl['y_delta_position'].cumsum()

butter_lw['y_delta_position'] = butter_lw['y_velocity'] * (butter_lw['time'] - butter_lw['time'].shift(1))
butter_lw['y_delta_position'][0] = 0
butter_lw['y_position'] = butter_lw['y_delta_position'].cumsum()

butter_rw['y_delta_position'] = butter_rw['y_velocity'] * (butter_rw['time'] - butter_rw['time'].shift(1))
butter_rw['y_delta_position'][0] = 0
butter_rw['y_position'] = butter_rw['y_delta_position'].cumsum()

# calculate z position
butter_ls['z_delta_position'] = butter_ls['z_velocity'] * (butter_ls['time'] - butter_ls['time'].shift(1))
butter_ls['z_delta_position'][0] = 0
butter_ls['z_position'] = butter_ls['z_delta_position'].cumsum()

butter_rs['z_delta_position'] = butter_rs['z_velocity'] * (butter_rs['time'] - butter_rs['time'].shift(1))
butter_rs['z_delta_position'][0] = 0
butter_rs['z_position'] = butter_rs['z_delta_position'].cumsum()

butter_ld['z_delta_position'] = butter_ld['z_velocity'] * (butter_ld['time'] - butter_ld['time'].shift(1))
butter_ld['z_delta_position'][0] = 0
butter_ld['z_position'] = butter_ld['z_delta_position'].cumsum()

butter_rd['z_delta_position'] = butter_rd['z_velocity'] * (butter_rd['time'] - butter_rd['time'].shift(1))
butter_rd['z_delta_position'][0] = 0
butter_rd['z_position'] = butter_rd['z_delta_position'].cumsum()

butter_ll['z_delta_position'] = butter_ll['z_velocity'] * (butter_ll['time'] - butter_ll['time'].shift(1))
butter_ll['z_delta_position'][0] = 0
butter_ll['z_position'] = butter_ll['z_delta_position'].cumsum()

butter_rl['z_delta_position'] = butter_rl['z_velocity'] * (butter_rl['time'] - butter_rl['time'].shift(1))
butter_rl['z_delta_position'][0] = 0
butter_rl['z_position'] = butter_rl['z_delta_position'].cumsum()

butter_lw['z_delta_position'] = butter_lw['z_velocity'] * (butter_lw['time'] - butter_lw['time'].shift(1))
butter_lw['z_delta_position'][0] = 0
butter_lw['z_position'] = butter_lw['z_delta_position'].cumsum()

butter_rw['z_delta_position'] = butter_rw['z_velocity'] * (butter_rw['time'] - butter_rw['time'].shift(1))
butter_rw['z_delta_position'][0] = 0
butter_rw['z_position'] = butter_rw['z_delta_position'].cumsum()

# cleanup
butter_ls = butter_ls.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_rs = butter_rs.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_ld = butter_ld.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_rd = butter_rd.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_ll = butter_ll.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_rl = butter_rl.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_lw = butter_lw.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])
butter_rw = butter_rw.drop(columns=['x_delta_velocity', 'y_delta_velocity', 'z_delta_velocity', 'x_delta_position', 'y_delta_position', 'z_delta_position'])

# example plot of acceleration, velocity, and position 
plt.figure(figsize=(12, 8))
plt.plot(butter_ls['time'], butter_ls['x'], alpha=0.5)
plt.plot(butter_ls['time'], butter_ls['x_velocity'], alpha=0.8)
plt.plot(butter_ls['time'], butter_ls['x_position'], alpha=0.8)
plt.legend(['accel', 'vel', 'pos'])
plt.xlabel('Time (seconds)')
plt.ylabel('m/s^2, m/s, m')
plt.savefig('output/ls.png')

plt.figure(figsize=(12, 8))
plt.plot(butter_lw['time'], butter_lw['x'], alpha=0.5)
plt.plot(butter_lw['time'], butter_lw['x_velocity'], alpha=0.8)
plt.plot(butter_lw['time'], butter_lw['x_position'], alpha=0.8)
plt.legend(['accel', 'vel', 'pos'])
plt.xlabel('Time (seconds)')
plt.ylabel('m/s^2, m/s, m')
plt.savefig('output/lw.png')

plt.figure(figsize=(12, 8))
plt.plot(butter_ll['time'], butter_ll['x'], alpha=0.5)
plt.plot(butter_ll['time'], butter_ll['x_velocity'], alpha=0.8)
plt.plot(butter_ll['time'], butter_ll['x_position'], alpha=0.8)
plt.legend(['accel', 'vel', 'pos'])
plt.xlabel('Time (seconds)')
plt.ylabel('m/s^2, m/s, m')
plt.savefig('output/ll.png')

plt.figure(figsize=(12, 8))
plt.plot(butter_rs['time'], butter_rs['x'], alpha=0.5)
plt.plot(butter_rs['time'], butter_rs['x_velocity'], alpha=0.8)
plt.plot(butter_rs['time'], butter_rs['x_position'], alpha=0.8)
plt.legend(['accel', 'vel', 'pos'])
plt.xlabel('Time (seconds)')
plt.ylabel('m/s^2, m/s, m')
plt.savefig('output/rs.png')

plt.figure(figsize=(12, 8))
plt.plot(butter_rw['time'], butter_rw['x'], alpha=0.5)
plt.plot(butter_rw['time'], butter_rw['x_velocity'], alpha=0.8)
plt.plot(butter_rw['time'], butter_rw['x_position'], alpha=0.8)
plt.legend(['accel', 'vel', 'pos'])
plt.xlabel('Time (seconds)')
plt.ylabel('m/s^2, m/s, m')
plt.savefig('output/rw.png')

plt.figure(figsize=(12, 8))
plt.plot(butter_rl['time'], butter_rl['x'], alpha=0.5)
plt.plot(butter_rl['time'], butter_rl['x_velocity'], alpha=0.8)
plt.plot(butter_rl['time'], butter_rl['x_position'], alpha=0.8)
plt.legend(['accel', 'vel', 'pos'])
plt.xlabel('Time (seconds)')
plt.ylabel('m/s^2, m/s, m')
plt.savefig('output/rl.png')