import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pykalman import KalmanFilter


data = pd.read_csv('../data/left_leg_standard.csv', header=0, names=['time', 'x', 'y', 'z', 't'])
data = data.drop(columns=['t'])
print(data)

# loess smoothing
lowess = sm.nonparametric.lowess
loess_smoothed = lowess(data['x'], data['time'], frac=0.001)

# kalman data filtering
kalman_data = data[['x', 'y', 'z']]
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
kalman_smoothed, covariances= kf.smooth(kalman_data)

plt.plot(data['time'], data['x'])
# plt.plot(data['time'], loess_smoothed[:,1])
plt.plot(data['time'], kalman_smoothed[:,0])
# plt.plot(data['time'], data['y'])
# plt.plot(data['time'], data['z'])
plt.show()