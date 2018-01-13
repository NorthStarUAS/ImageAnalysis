# Notes ...

## Optimizer

   This preps the data for use with a scipy least squares optimizer.
   I provide a jacobian 'sparcity' matrix to take performance
   advantage of the sparse nature of this optimization problem.  At
   the time of this writing, the scipy optimzer is working far more
   robustly (stably) than the Lourakis optimzer with my own data
   sets.  In addition, moving to the scipy optimizer standardizes the
   code, and eliminates an external 3rd party dependency.

   Many optimizers I've seen are setup to optimize for per-image
   camera calibratoin and distortion parameters.  This optimizer is
   setup to assume all images are taken with the same camera so we can
   globally optimize for a single camera calibration matrix and single
   set of distortion parameters.  This seems to be more productive
   than solving for individual calibrations.
   
   Keeping focal length bounded to +/- 5% of original estimate also
   seems productive.