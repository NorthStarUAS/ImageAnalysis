# Install ImageAnalysis source code package for devel and testing purposes

This document is intended for those that are interested in running the
native python scripts on their own computer.  This doc a work and
progress and initially focused on the windows platform.  The
essentials are similar for any other platform so this guide may still
be helpful for Mac and Linux users (who may have many of these
components already installed by default on their systems.)  Note: I do
all my development and testing on Linux.

The following install recipe ensures all the required prerequisites
are in place for running the ImageAnalysis software:

# Install Python3 (Anaconda Python 3.x version)

  Download/install: https://www.anaconda.com/distribution

  Go with all the defaults (unless you are a python setup expert and
  have a specific reason to do otherwise.)

  Do not Install Microsoft VSCode (Skip!) unless you really want it for
  some other project not related to this one.

# Install git for windows

  Download/install: https://git-scm.com/download/win

  (Again my advice is to go with all the defaults choices the
  installer presents.)

# Open command shells

  - Open an Anaconda3 -> Anaconda Prompt (from the start menu)
  - Open a git bash shell (from the start menu)

# Install required Python system packages

  From the Anaconda Prompt:

    anaconda3> conda install opencv
    anaconda3> conda install progress
    anaconda3> pip install Panda3D
    anaconda3> pip install geojson

# Install additional required software packages

  From the Bash prompt:

    bash$ cd Desktop
    bash$ mkdir Software
    bash$ cd Software

  # Install NavPy:

    bash$ cd Desktop/Software
    bash$ git clone https://github.com/NavPy/NavPy.git
    anaconda3> cd Desktop\Software\NavPy
    anaconda3> python setup.py install
  
  # Install pyprops:

    bash$ cd Desktop/Software
    bash$ git clone https://github.com/AuraUAS/aura-props.git
    anaconda3> cd Desktop\Software\aura-props\python
    anaconda3> python setup.py install
  
# Install Image Analysis

    bash$ git clone https://github.com/UASLab/ImageAnalysis.git

# Run the visualizer script

    anaconda3> cd Desktop\Software\ImageAnalysis\scripts
    anaconda3> python 7a-python.py