# Instructions for explorer.py used to view maps

## Installation instructions: explorer.py

Installation of the map visualization tool (explorer.py) for windows is very
simple, but does take a few minutes to download and it helps if you scan through
this document and understand the basics of the process before starting.

Quick overview: The explorer.py is a map viewer than runs on your own computer
and visualizes aerial survey data sets that stored locally on your hard drive.
The mani advantage over cloud based systems are:

* The software and tool chain are free to use and modify -- licensed under the
  MIT open-source license.

* You have direct access to all the original images to see in their full
  resolution.  The visualizer can apply a filter to the images to make the
  details or colors pop out better.  You can cycle between the different images
  that cover your point of focus to see the different angles.  All images are
  shown seamlessly in their correct "map" location and orientation.

## System requirements

* You need a windows computer to run the downloadable version.  (Or any other
  cmputer if you want to setup all the python3 dependencies yourself.)
* Enough hard drive space to store your image data sets locally (this can add up
  quickly, an external drive might be your friend.)

## Usage Instructions

* Download the following zip file to anywhere (like your desktop):
  <https://github.com/UASLab/ImageAnalysis/releases/download/v20190215/7a-explore.zip>

* Extract this zip file to a folder.  (After extracting, you can delete the
  original zip file if you wish.)  It shouldn't matter where this folder lives
  as long as you know where to find it.

* Inside the new folder is a whole bunch of files which you can ignore.  Skip
  all the files and scroll to (or find) the application called "7a-explore".
  Just double click on it to run it and it will ask you where the data set is.
  (You won't have a data set at this point.)

* If you can run 7a-explore and get to the point where it asks you for a data
  set, you should be good to go.

## Grab a data set.  It should be almost the exact same process

* Here is a 'small' data set if you want to test the software before committing
  to downloading the entire elm creek data set. This data set is only 3 Gb and
  again, you can download it to anywhere on your computer that works for you.
  Here is the link:
  <https://drive.google.com/open?id=1MA-kt5hHmbX290M59KI0pBwV-bA6GVPs>

* Extract this .zip file (you can then delete the original zip file if you want
  to save space on your hard drive.)  This should create a folder called
  'avon-park'.

* Here is the link to download the elm creek data set if you want to just jump
  straight to that (25Gb):
  <https://drive.google.com/open?id=1fE02t4SJKeAKej9dLCZH0DSjzNeNIpOk>

  The data sets can be huge, but this is kind of the point, you get access to
  all the original images in their full resolution, and all the different angles
  that cover every spot.  For those of us searching for needles in the haystack
  (I think) this can be helpful.

## Navigation

When the map comes up you can use your mouse to pan and zoom around kind of like
a google map.  The system will load the high res version of each texture as you
move so each time you move, just give the system a half second to load the new
high-res texture before trying to move again.  You can also use arrow keys to
pan around and +/- keys to zoom in and out.

## Annotations

You can use a right mouse click to drop down an annotation (or right click on an
existing marker to delete it.)  The system will update a file called
annotations.csv in the data directory ... hopefully this will eventually allow
us to import our annotations into other software like EDDmaps.

## Selecting different pictures for your center of view

One nice thing my software can do that drone deploy cannot is it can show you
all the different views that over lap your center of focus. Seeing the object
from various angles can often be super helpful for identifying what it is (or
isn't.)  You can select alternative views with the number keys: 0 (default), 1,
2, 3, 4, 5, ..., 9 As soon as you move/pan again, you'll flip back to the
default image.

## Oops?

If something goes wrong or totally doesn't make sense, please let me
know!  If you see a bunch of big black trapezoids near your center of
view, I can fix that, but I need you to send me some debug info from
you computer to help me with that.
