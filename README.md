# cmpt353_project
If you compare right-foot vs left-foot, can you determine if someone's gait is asymmetrical? Perhaps this can be used to detect an injury. (This would probably be easier if you have somebody with an injury to work with. Please do not injure your friends to answer this question.)

## Interpreting the Data
phone is attached to the ankle facing outwards (add pic?)

x: leg moving forward/backwarad

y: leg moving up/down

z: leg moving side to side

## TODO
-get datasets: normal, limp, dragging feet

-smooth/clean data (butterworth filter)

-get frequency of steps (FFT, fourier transform)

-transform acceleration to velocity to position (Δv = a⋅Δt and Δp = v⋅Δt.)

-check normality/variance 

-statistical test to see if datasets differ (anova, u-test, t-test)

-compare different steps

-train ml to detect injury


-plots: 

..* raw data (before/after smoothing)

..* velocity vs time

..* position vs time

..* graph of foot movement? (side view, top view, front view)
