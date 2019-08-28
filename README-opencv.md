Well, every time I look, opencv/fedora changes how things are
packaged.  The current state (Dec 04, 2018) is that we have to build
ourselves from source code in order to get access to the SIFT
functionality.  Another year and a half a the SIFT patent expires,
maybe it can become part of the base opencv at that point?

Fedora installation tips

# required packages

  - dnf install gtk3-devel libdc1394-devel libv4l-devel ffmpeg ffmpeg-devel gstreamer-plugins-base-devel
  - dnf install cmake ccache g++ openblas-devel atlas-devel python3-h5py hdf5-devel gflags-devel glog-devel tesseract-devel vtk-devel

  - ? gtk2-devel ?

# recommended packages

  - dnf install libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel libtiff-devel libwebp-devel
  - dnf install tbb-devel eigen3-devel doxygen openblas-devel hdf5-devel gflags-devel glog-devel vtk-devel
  - dnf install python3-numpy python3-scipy ccache python3-matplotlib python3-piexif python3-tqdm python3-geojson python3-xmp-toolkit

# download the latest stable .zip from here:

  https://sourceforge.net/projects/opencvlibrary/files

# extract
$ cd into source tree

# contrib modules
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv_contrib
$ git checkout X.Y.Z (version matches the opencv version we are building)

$ cd back to opencv-X.Y.Z
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_TBB=ON -DWITH_EIGEN=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DOPENCV_ENABLE_NONFREE=ON -DPYTHON_DEFAULT_EXECUTABLE=python3 ..

# build it:
$ make -j4

# install it:
$ sudo make install

# make it accessible to yourself by adding this to your ~/.bashrc file:
# and if you run [t]csh or something else you probably already know how to
# do the equivalent in your own shell.

  # User specific aliases and functions
  export PYTHONPATH=$PYTHONPATH:/usr/local/python/cv2/python-3.7

