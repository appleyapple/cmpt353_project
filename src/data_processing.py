import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pykalman import KalmanFilter


def load_data(csv):
    data = pd.read_csv('data/'+csv, header=0, names=['time', 'x', 'y', 'z', 't'])
    data = data.drop(columns=['t'])
    return data

# data loading
left_standard = load_data('left_leg_standard.csv')
right_standard = load_data('right_leg_standard.csv')

# loess smoothing
lowess = sm.nonparametric.lowess
left_standard_loess = lowess(left_standard['x'], left_standard['time'], frac=0.001)

# kalman data filtering
kalman_data = left_standard[['x', 'y', 'z']]
initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([0.98, 0.88, 0.83]) ** 2 
transition_covariance = np.diag([0.005, 0.01, 0.01]) ** 2
transition = [[0.97,0.5,0.2], [0.1,0.4,2.2], [0,0,0.95]] 

# kalman smoothing
kf = KalmanFilter(
    transition_matrices=transition,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance,
    initial_state_mean=initial_state
)
left_standard_kalman, covariances= kf.smooth(kalman_data)

plt.plot(left_standard['time'], left_standard['x'])
plt.plot(left_standard['time'], left_standard_loess[:,1])
# plt.plot(left_standard['time'], left_standard_kalman[:,0])
# plt.plot(data['time'], data['y'])
# plt.plot(data['time'], data['z'])
plt.show()