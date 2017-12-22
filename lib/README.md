# Notes ...

## SBA1 (deprecated)

   This is the 'original' python front end the prep'd the data, called
   an external optimizer, and extracted the results.  The external
   optimizer is SBA v1.6 by Manolis Lourakis - Institute of Computer
   Science - Foundation for Research and Technology - Hellas -
   Heraklion, Crete, Greece.

   Note I struggled with some stability and convergence issues with
   this solver that seemed to be improved when I switched to the scipy
   sparse optimizer.  I never determined if there were issues with how
   I prepared the data, issues with my invocation and settings, or
   issues with the optimizer itself.  Chances are, the problems were
   on my side, but I could never track them down.

## SBA2 (current)

   This preps the data for use with a scipy least squares optimizer.
   I provide a jacobian 'sparcity' matrix to take performance
   advantage of the sparse nature of this optimization problem.  At
   the time of this writing, the scipy optimzer is working far more
   robustly (stablely) than the Lourakis optimzer with my own data
   sets.  In addition, moving to the scipy optimizer standardizes the
   code, and eliminates an external 3rd party dependency.