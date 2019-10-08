#!/usr/bin/python3

import argparse
import fnmatch
import os
import re
import shutil

parser = argparse.ArgumentParser(description="Copy files while numerically adding a constant value to the file name numbering scheme.")
parser.add_argument('--src', required=True, help="image source directory (i.e. from the SD card")
parser.add_argument('--dest', required=True, help="image destination directory")
parser.add_argument('--add', required=True, type=int,
                    help="add this to file name number")
args = parser.parse_args()

# scan src dir
files = []
for file in os.listdir(args.src):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        print(file)
        dest_name = os.path.join(args.dest, file)
        m = re.search('(.+)_(\d+)\.(.+)', file)
        #print(m.group(0), m.group(1), m.group(2), m.group(3))
        num =  int(m.group(2)) + args.add
        new_file = "%s_%04d.%s" % (m.group(1), num, m.group(3))
        dest_name = os.path.join(args.dest, new_file)
        src_name = os.path.join(args.src, file)
        if os.path.exists(dest_name):
            print("")
            print(dest_name, "exists!")
            print("")
            print("ABORTING!")
            quit()
        print("cp:", src_name, dest_name)
        shutil.copy2(src_name, dest_name)
