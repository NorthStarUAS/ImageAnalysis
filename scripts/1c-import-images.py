#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import ProjectMgr

# for all the images in source_dir, scale them and write the scaled
# version to dest_dir.
#
# converter = 'imagemagick' uses the 'convert' command.  This should
#             preserve image meta data (exif, etc.)
#
# converter = 'opencv' uses opencv commands to resize the image.  This
#             is much faster than the external convert command, but
#             does not preserve any meta data.

parser = argparse.ArgumentParser(description='Import a directory of images into a project.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--source', required=True, help='image source directory')
parser.add_argument('--scale', type=float, default=0.25, help='scale factor')
parser.add_argument('--converter', default='imagemagick',
                    choices=['imagemagick', 'opencv'])

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.set_source_dir( args.source )
proj.import_images( scale=args.scale, converter=args.converter)
proj.save()
