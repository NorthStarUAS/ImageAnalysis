# 1. Project Setup

## 1a-create-project.py

Create a project directory and stub config files

## 1b-set-camera-config.py

Define camera calibration and lens distortion parameters.

## 1c-import-images.py

Import images into project work space (may optionally scale the images for
faster processing and less memory/disk space usage.)

## 1d-load-images.py

Tests loading of image info, then ends.

## 1e-show-images.py

Show all the imported images.


# 2. Set Initial Camera Poses

# 3. Feature Detection

# 4. Feature Matching


# 5. Assemble Scene / Bundle Adjustment

## 5a-sba1.py

Original attempt, deprecated in favor of 5a-sba2.py

## 5a-sba2.py

Takes initial direct georeference scene layout and runs bundle adjustment.

## 5a-sba3.py

Uses image connectivity structure combined with solvepnp() and a
custom multi-vector triangulation function to assemble scene.  Works
well until we run out of sufficient connectivity.

## 5a-sba4.py

Test of opencv3's findEssentialMat() and recoverPose() to build the
initial scenery construction.


# 6. Render Results