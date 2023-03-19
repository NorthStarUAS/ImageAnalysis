#!/usr/bin/env python3

# This is a quick hack script to migrate some old image sets and
# flight data forward to work with the newest code.  It's not
# generally useful for new projects or new users.  I will even
# hard-code paths to make this more clear. :-)

# this creates a pix4d.csv file from a collection of old *.info files

import csv
import fnmatch
import os

# from the aura-props package
from props import PropertyNode
import props_json

source_dir = "/home/curt/Projects/ImageAnalysis/scripts/avon-park2/Images"
output_file = 'pix4d.csv'

# scan the source dir for *.info files
db = {}
for file in sorted(os.listdir(source_dir)):
    if fnmatch.fnmatch(file, '*.info'):
        name, ext = os.path.splitext(file)
        info_node = PropertyNode()
        props_json.load(os.path.join(source_dir, file), info_node)
        print(name + ".JPG")
        db[name + '.JPG'] = info_node

with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter( csvfile,
                             fieldnames=['File Name',
                                         'Lat (decimal degrees)',
                                         'Lon (decimal degrees)',
                                         'Alt (meters MSL)',
                                         'Roll (decimal degrees)',
                                         'Pitch (decimal degrees)',
                                         'Yaw (decimal degrees)'] )
    writer.writeheader()
    for image_name, info_node in db.items():
        print(image_name)
        pose_node = info_node.getChild('aircraft-pose')
        print(pose_node.getFloatEnum('lla', 0))
        print(pose_node.getFloatEnum('lla', 1))
        lat_deg = pose_node.getFloatEnum('lla', 0)
        lon_deg = pose_node.getFloatEnum('lla', 1)
        alt_m = pose_node.getFloatEnum('lla', 2)
        yaw_deg = pose_node.getFloatEnum('ypr', 0)
        if yaw_deg < 0: yaw_deg += 360.0
        pitch_deg = pose_node.getFloatEnum('ypr', 1)
        roll_deg = pose_node.getFloatEnum('ypr', 2)
        writer.writerow( { 'File Name': os.path.basename(image_name),
                           'Lat (decimal degrees)': "%.10f" % lat_deg,
                           'Lon (decimal degrees)': "%.10f" % lon_deg,
                           'Alt (meters MSL)': "%.2f" % alt_m,
                           'Roll (decimal degrees)': "%.2f" % roll_deg,
                           'Pitch (decimal degrees)': "%.2f" % pitch_deg,
                           'Yaw (decimal degrees)': "%.2f" % yaw_deg } )

print('Remember to move the ./pix4d.csv file to the original image directory')
