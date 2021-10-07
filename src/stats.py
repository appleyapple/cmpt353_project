import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.filter_design import butter
import statsmodels.api as sm
from scipy import signal
from scipy import stats
from scipy.fft import fft, fftfreq, fftshift
import matplotlib as mpl
from . import data_processing as data


# fft 
d = np.array(data.butter_ls['x'])
fft = fft(d)
freq = fftshift(fftfreq(d.shape[-1]))
#print(fft)

# https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
T = data.butter_ls['time'][1] - data.butter_ls['time'][0]
N = len(data.butter_ls['x'])
f = np.linspace(0, 1 / T, N)

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")    
plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)
# plt.show()

# attempt to transforming the data to normal distribution
plt.title('standard x-axis acceleration histogram untransformed and transformed')
plt.hist(data.butter_ls['x'])
plt.hist(np.sqrt(data.butter_ls['x']+1))
plt.hist(np.log(data.butter_ls['x']+1))
plt.legend(['Untransformed', 'Square-root Transformed','Log Transformed'])
plt.xlabel('Acceleration')
plt.ylabel('Count ')
#plt.show()

#given how there are negative values, square-root and logarithms wouldn't work so tried shifting values, but the shift overwhelms the original values so any statistical tests requiring normality cannot be done

# stats testing all the left leg data against right leg using Mann-Whitney Test since they are non-normal datasets
print("\nMann-Whitney Test For:\n")
print("X-axis")
print("Acceleration")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['x'],data.butter_rs['x']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['x'],data.butter_rd['x']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['x'],data.butter_rw['x']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['x'],data.butter_rl['x']).pvalue)+"\n")

print("Velocity")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['x_velocity'],data.butter_rs['x_velocity']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['x_velocity'],data.butter_rd['x_velocity']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['x_velocity'],data.butter_rw['x_velocity']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['x_velocity'],data.butter_rl['x_velocity']).pvalue)+"\n")

print("Position")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['x_position'],data.butter_rs['x_position']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['x_position'],data.butter_rd['x_position']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['x_position'],data.butter_rw['x_position']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['x_position'],data.butter_rl['x_position']).pvalue)+"\n")

print("\nY-axis")
print("Acceleration")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['y'],data.butter_rs['y']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['y'],data.butter_rd['y']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['y'],data.butter_rw['y']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['y'],data.butter_rl['y']).pvalue)+"\n")

print("Velocity")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['y_velocity'],data.butter_rs['y_velocity']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['y_velocity'],data.butter_rd['y_velocity']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['y_velocity'],data.butter_rw['y_velocity']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['y_velocity'],data.butter_rl['y_velocity']).pvalue)+"\n")

print("Position")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['y_position'],data.butter_rs['y_position']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['y_position'],data.butter_rd['y_position']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['y_position'],data.butter_rw['y_position']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['y_position'],data.butter_rl['y_position']).pvalue)+"\n")

print("\nZ-axis")
print("Acceleration")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['z'],data.butter_rs['z']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['z'],data.butter_rd['z']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['z'],data.butter_rw['z']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['z'],data.butter_rl['z']).pvalue)+"\n")

print("Velocity")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['z_velocity'],data.butter_rs['z_velocity']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['z_velocity'],data.butter_rd['z_velocity']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['z_velocity'],data.butter_rw['z_velocity']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['z_velocity'],data.butter_rl['z_velocity']).pvalue)+"\n")

print("Position")
print("Standard: " + str(stats.mannwhitneyu(data.butter_ls['z_position'],data.butter_rs['z_position']).pvalue))
print("Dragging: " + str(stats.mannwhitneyu(data.butter_ld['z_position'],data.butter_rd['z_position']).pvalue))
print("Waddle: " + str(stats.mannwhitneyu(data.butter_lw['z_position'],data.butter_rw['z_position']).pvalue))
print("Limp: " + str(stats.mannwhitneyu(data.butter_ll['z_position'],data.butter_rl['z_position']).pvalue)+"\n")

# conclusion from stats test is that  they are too sensitive to changes in data, so ML is probably more suited at discerning the differences in leg movement types



