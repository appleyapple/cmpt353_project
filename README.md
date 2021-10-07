# cmpt353_project
If you compare right-foot vs left-foot, can you determine if someone's gait is asymmetrical? Perhaps this can be used to detect an injury. (This would probably be easier if you have somebody with an injury to work with. Please do not injure your friends to answer this question.)

## Run code
### Required libraries:
scipy, pandas, numpy, matplotlib, statsmodels

### Command execution:
#### Run whole thing
`> python main.py`

### File outputs:
The first initial for each png indicates left(l) or right(r) and the second initial for the type of walk: standard(s), limp(l), waddle(w), drag(d) 

Output folder contains butterworth_filter and loess_smooth folders, as well as acceleration, velocity and position graphs for each walking type and a numerical integration test graph

Butterworth_filter and loess_smooth folders contain graphs of raw acceleration data vs noise filtered acceleration data

In the console progression updates and an output of p-values from the mann-whitney u test between the left leg and right leg for the same walking types

## Interpreting the Data
Phone is attached to the ankle facing outwards (add pic?)

x: leg moving forward/backwarad

y: leg moving up/down

z: leg moving side to side

## TODO
- [x] get datasets: normal, limp, dragging feet

- [x] smooth/clean data (crop data?, butterworth filter)

- [  ] get frequency of steps (FFT, fourier transform)

- [x] transform acceleration to velocity to position (Δv = a⋅Δt and Δp = v⋅Δt.)

- [x] check normality/variance (of the acceleration data)

- [x] statistical test to see if datasets differ (mann-whitney u)

- [  ] compare different steps

- [  ] train ml to detect injury


-plots: 

- [x] raw data & smoothing

- [x] velocity vs time

- [x] position vs time

- [  ] graph of foot movement? (side view, top view, front view)
