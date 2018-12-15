Well, every time I look, opencv/fedora changes how things are packaged.  The
current state (Dec 04, 2018) is that we have to build ourselves from source
code in order to get access to the SIFT functionality.

# required
# dnf install gtk2-devel libdc1394-devel libv4l-devel ffmpeg-devel gstreamer-plugins-base-devel

# optional
# dnf install libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel libtiff-devel libwebp-devel

# also good to have
dnf install tbb-devel eigen3-devel doxygen

# download the latest stable .tar.gz from here:
https://sourceforge.net/projects/opencvlibrary/files

# extract
$ cd into source tree
$ mkdir build
$ cd build

# contrib modules
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv_contrib
$ git checkout X.Y.Z (version matches the opencv version we are building)

# cd back to opencv-opencv-blah-blah/build
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

