# Date: July 13, 2021

# Install Python3 (Anaconda Python 3.x version)

  Download & install: https://www.anaconda.com/products/individual

  I recomment not changing any of the install defaults (unless you are
  are familiar with python setup choices and have a specific reason to
  do something different.)

# Download Image Analysis tools

  browser -> https://github.com/UASLab/ImageAnalysis/
  Code -> download zip (ImageAnalysis-master.zip)
  Extract All -> to Desktop (or your preferred location)
  
# Setup the Anaconda3 environment

  Start Menu -> Anaconda3 -> Anaconda Prompt:

  > cd to the ImageAnalysis install location (from previous step).
    Make sure you are inside the top ImageAnalysis-master directory.
  > There should be an "environment.yml" file here.
  > Run: "conda env create -f environment.yml"
    This will install all the libraries and prerequisites to run the
    image analysis tools.

# Activate the environment

  Do this each time you start an anaconda prompt to run the
  ImageAnalysis tools.

  (From an anaconda3 prompt)

  Run: "conda activate ImageAnalysis"

# Run the explorer.py tool

  (From an anaconda3 prompt)

  > cd to the ImageAnalysis install location (from previous step).
    Make sure you are inside the top ImageAnalysis-master directory.

  > cd to the scripts subdirectory one level below the main install
    directory.

  > run: "python explorer.py"

    After a few moments, a folder selector box will open up and you
    can select the folder containing the data set you wish to explore.

# END OF INSTRUCTIONS

The remainder of this file is just odds and ends of commands and notes
to self.  You shouldn't need to run any of these commands yourself.

> conda create --name imageanalysis python pip opencv scipy tqdm git
> conda activate imageanalysis

## Install additional 3rd party packages

> pip install panda3d simplekml geojson
> pip install git+https://github.com/RiceCreekUAS/props.git/#subdirectory=python
> pip install git+https://github.com/NavPy/NavPy.git

# install 


  browser -> https://github.com/RiceCreekUAS/props
  Code -> download zip (NavPy-master.zip)
  Extract All -> to Desktop
  open anaconda prompt
  > cd Desktop\NavPy-master
  > python setup.py install
  
  Browser -> https://github.com/NavPy/NavPy
  Code -> download zip (props-master.zip)
  Extract All -> to Desktop
  open anaconda prompt
  > cd Desktop\props-master\python
  > python setup.py install


=========
> conda create --name imageanalysis python pip opencv scipy tqdm git
> conda activate imageanalysis
> conda deactivate

> conda install opencv
> conda upgrade
  ﻿

conda env export > environment.yml
- manually edit pip dependencies to list github path, not just package
  name where needed.
conda env create -f environment.yml