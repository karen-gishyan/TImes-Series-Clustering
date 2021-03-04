### To do
[X] Understand the metric function from 
https://github.com/tslearn-team/tslearn/blob/a3cf3bf/tslearn/clustering/utils.py#L66-L197
* if not string or none, assumes a callable for distance calculation.
[ ] Understand 
https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.cdist_dtw.html#tslearn.metrics.cdist_dtw
[ ] The silhouettes match the results (check one more time), but check the plots, to  make sure the right variables are correctly displayed next to each plot.
[ ] Restructure the code, complete readme.
[ ] Next time avoid using pandas as much as possible (work with numpy), and work with fig and ax,
    not strictly with plot.
[ ] Include the results for dtw and euclidean.
[ ] Site tslearn, maybe yellow-brick as well.
[x] Add x and y series for the clustering plot.
[x] Manually check if there are clusters which are empty for clusters which only have a blue line.
[ ] Update the examples in time-series.py the stationarity check.
[ ] Try to include a generator,
[ ] include typing and proper docstring.
[ ] make a final check if the labels of the series are correct.
