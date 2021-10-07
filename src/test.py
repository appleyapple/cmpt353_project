import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
import matplotlib as mpl
import scipy.integrate as integrate
import scipy.interpolate
from scipy import stats

#to remove a figure producing limit warning
mpl.rcParams['figure.max_open_warning'] = 0

def load_data(csv):
    data = pd.read_csv('data/'+csv, header=0, names=['time', 'x', 'y', 'z', 't'])
    data = data.drop(columns=['t'])
    return data

print("Integration Test Starting")

# data loading
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

# loess smoothing
loess_smoothed_datasets_x = []
loess_smoothed_datasets_y = []
loess_smoothed_datasets_z = []
lowess = sm.nonparametric.lowess

#for x-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = lowess(dataset['x'], dataset['time'], frac=0.025)
    loess_smoothed_datasets_x.append(smoothed_dataset)


#for y-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = lowess(dataset['y'], dataset['time'], frac=0.025)
    loess_smoothed_datasets_y.append(smoothed_dataset)


#for z-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = lowess(dataset['z'], dataset['time'], frac=0.025)
    loess_smoothed_datasets_z.append(smoothed_dataset)

# butterworth filter
butter_smoothed_datasets_x = []
butter_smoothed_datasets_y = []
butter_smoothed_datasets_z = []
b,a = signal.butter(5, 0.012, btype='lowpass', analog=False)

#for x-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = signal.filtfilt(b, a, dataset['x'])
    butter_smoothed_datasets_x.append(smoothed_dataset)

#for y-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = signal.filtfilt(b, a, dataset['y'])
    butter_smoothed_datasets_y.append(smoothed_dataset)


#for z-axis
for dataset, name in zip(datasets, datasets_str):
    smoothed_dataset = signal.filtfilt(b, a, dataset['z'])
    butter_smoothed_datasets_z.append(smoothed_dataset)

#butterworth dataframes ls = left standard, ld = left drag, ll = left limp, lw = left waddle, etc..
butter_ls = pd.DataFrame({'time':loess_smoothed_datasets_x[0][:,0],'left_standard_x':butter_smoothed_datasets_x[0], 'left_standard_y':butter_smoothed_datasets_y[0],'left_standard_z':butter_smoothed_datasets_z[0]})
butter_rs = pd.DataFrame({'time':loess_smoothed_datasets_x[1][:,0],'right_standard_x':butter_smoothed_datasets_x[1], 'right_standard_y':butter_smoothed_datasets_y[1],'right_standard_z':butter_smoothed_datasets_z[1]}) 
butter_ld = pd.DataFrame({'time':loess_smoothed_datasets_x[2][:,0],'left_drag_x':butter_smoothed_datasets_x[2], 'left_drag_y':butter_smoothed_datasets_y[2],'left_drag_z':butter_smoothed_datasets_z[2]})
butter_rd = pd.DataFrame({'time':loess_smoothed_datasets_x[3][:,0],'right_drag_x':butter_smoothed_datasets_x[3], 'right_drag_y':butter_smoothed_datasets_y[3],'right_drag_z':butter_smoothed_datasets_z[3]})
butter_ll = pd.DataFrame({'time':loess_smoothed_datasets_x[4][:,0],'left_limp_x':butter_smoothed_datasets_x[4], 'left_limp_y':butter_smoothed_datasets_y[4],'left_limp_z':butter_smoothed_datasets_z[4]})
butter_rl = pd.DataFrame({'time':loess_smoothed_datasets_x[5][:,0],'right_limp_x':butter_smoothed_datasets_x[5], 'right_limp_y':butter_smoothed_datasets_y[5],'right_limp_z':butter_smoothed_datasets_z[5]}) 
butter_lw = pd.DataFrame({'time':loess_smoothed_datasets_x[6][:,0],'left_waddle_x':butter_smoothed_datasets_x[6], 'left_waddle_y':butter_smoothed_datasets_y[6],'left_waddle_z':butter_smoothed_datasets_z[6]}) 
butter_rw = pd.DataFrame({'time':loess_smoothed_datasets_x[7][:,0],'right_waddle_x':butter_smoothed_datasets_x[7], 'right_waddle_y':butter_smoothed_datasets_y[7],'right_waddle_z':butter_smoothed_datasets_z[7]})

#numerical integration to find velocity and position

#adding velocity and position
datasets_str = ['left_standard', 'right_standard', 'left_drag', 'right_drag', 'left_limp', 'right_limp', 'left_waddle', 'right_waddle']
butter_datasets = [butter_ls, butter_rs, butter_ld, butter_rd, butter_ll, butter_rl, butter_lw, butter_rw]

#functions for integrating dataframe information
def integ_vel(df, t):
    if t == 'x':
        y = df.iloc[5]
    elif t == 'y':
        y = df.iloc[6]
    else: 
        y = df.iloc[7]
    dx = df.iloc[4]
    return np.trapz(y=y, dx=dx)

def integ_pos(df, t):
    if t == 'x':
        y = df.iloc[14]
    elif t == 'y':
        y = df.iloc[15]
    else: 
        y = df.iloc[16]
    dx = df.iloc[4]
    return np.trapz(y=y, dx=dx)


#preparing data for velocity integration by adding a dx column and y-pair columns
for data,name in zip(butter_datasets,datasets_str):
    data['dx'] = data['time']-data['time'].shift(1)
    data['dx'].iloc[0] = 0

    data['pair_vel_x'] = list(zip(data[name+'_x'].shift(1),data[name+'_x']))
    data['pair_vel_x'].iloc[0] = (0,data[name+'_x'].iloc[0])

    data['pair_vel_y'] = list(zip(data[name+'_y'].shift(1),data[name+'_y']))
    data['pair_vel_y'].iloc[0] = (0,data[name+'_y'].iloc[0])

    data['pair_vel_z'] = list(zip(data[name+'_z'].shift(1),data[name+'_z']))
    data['pair_vel_z'].iloc[0] = (0,data[name+'_z'].iloc[0])

#velocity integration
for data in butter_datasets:
    data['d_vel_x'] = data.apply(integ_vel,axis=1,t='x')
    data['vel_x'] = data['d_vel_x'].cumsum()

    data['d_vel_y'] = data.apply(integ_vel,axis=1,t='y')
    data['vel_y'] = data['d_vel_y'].cumsum()

    data['d_vel_z'] = data.apply(integ_vel,axis=1,t='z')
    data['vel_z'] = data['d_vel_z'].cumsum()

#preparing data for position integration
for data in butter_datasets:
    data['pair_pos_x'] = list(zip(data['vel_x'].shift(1),data['vel_x']))
    data['pair_pos_x'].iloc[0] = (0,data['vel_x'].iloc[0])

    data['pair_pos_y'] = list(zip(data['vel_y'].shift(1),data['vel_y']))
    data['pair_pos_y'].iloc[0] = (0,data['vel_y'].iloc[0])

    data['pair_pos_z'] = list(zip(data['vel_z'].shift(1),data['vel_z']))
    data['pair_pos_z'].iloc[0] = (0,data['vel_z'].iloc[0])

#position integration
for data in butter_datasets:
    data['d_pos_x'] = data.apply(integ_pos,axis=1,t='x')
    data['pos_x'] = data['d_pos_x'].cumsum()

    data['d_pos_y'] = data.apply(integ_pos,axis=1,t='y')
    data['pos_y'] = data['d_pos_y'].cumsum()

    data['d_pos_z'] = data.apply(integ_pos,axis=1,t='z')
    data['pos_z'] = data['d_pos_z'].cumsum()

#example plot used in the report
plt.figure(figsize=(12, 8))
plt.title('standard x-axis acceleration, velocity, position')
plt.plot(butter_ls['time'], butter_ls['left_standard_x'], alpha=0.8)
plt.plot(butter_ls['time'], butter_ls['vel_x'], alpha=0.8)
plt.plot(butter_ls['time'], butter_ls['pos_x'], alpha=0.8)
plt.legend(['Right Leg', 'Left Leg'])
plt.xlabel('Time (seconds)')
plt.ylabel('x ')
plt.savefig('output/ll_int_test')

