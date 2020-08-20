#!/usr/bin/env python3

import argparse
import fnmatch
import os
import re
import shutil

parser = argparse.ArgumentParser(description='Add a value to the file name for numbered files (renames the file with a new number).')
parser.add_argument('--add', required=True, type=int,
                    help='add this to file name number')
parser.add_argument('files', metavar='files', nargs='+',
                    help='list of files to modify')
args = parser.parse_args()

# scan src dir
files = []
for file in args.files:
    print(file)
    dirname = os.path.dirname(file)
    basename = os.path.basename(file)
    print("dir:", dirname)
    print("base:", basename)
    m = re.search('(\D*)(\d+)\.(.+)', basename)
    print(m.group(0), ":", m.group(1), m.group(2), m.group(3))
    new_num =  "%d" % (int(m.group(2)) + args.add)
    while len(new_num) < len(m.group(2)):
        new_num = '0' + new_num
    new_base = "%s%s.%s" % (m.group(1), new_num, m.group(3))
    print("rename:", file, os.path.join(dirname, new_base))
    os.rename(file, os.path.join(dirname, new_base))
