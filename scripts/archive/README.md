This is where old scripts go to work on all the projects they wanted
to do all their lives, but never had time due to kids and career.

## 4d-reset-matches-ned1.py

   start with all the original pair-wise matches, assemble them into a
   unified match structure and estimate their 3d location. (earlier
   revision.)

## 4d-reset-matches-ned2.py

   start with all the original pair-wise matches, assemble them into a
   unified match structure and estimate their 3d
   location. (intermediate revision.)

## 4z-collapse-matches.py

   similar to the reset-matches scripts, but also attempts to do full
   feature chain match grouping.  The jury is still out on whether
   this code is correct, or if it over constrains the problem and
   causes trouble for the optimizer.

## 5a-sba1.py

   Original attempt, deprecated in favor of 5a-sba2.py

## 5a-sba3.py

   Uses image connectivity structure combined with the ransac version
   of solvepnp() and a custom multi-vector triangulation function to
   assemble scene.  Seems to produce a useful initial scene structure
   for the set of images that are well connected together.  The ransac
   version of solvepnp() helps the robustness of the assembly process.

## 5a-sba4.py

   Test of opencv3's findEssentialMat() and recoverPose() to build the
   initial scenery construction.  Given all the pairwise relative
   rotations, send to optimizer to find an optimal set of camera
   orientations that minimize the slop.

