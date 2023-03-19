#!/usr/bin/env python3

# Import CSV annotations from Christina

import argparse
import csv
import json
import os

parser = argparse.ArgumentParser(description="Lookup a weather report for the location/time an image was captured.")
parser.add_argument("project", help="project directory")
parser.add_argument("csv_file", help="source csv file")
args = parser.parse_args()

with open(args.csv_file) as csvfile:
    reader = csv.DictReader(csvfile)
    markers = []
    for row in reader:
        lat_deg = None
        lon_deg = None
        alt_m = None
        id = None
        for key in row.keys():
            if "latitude" in key.lower():
                lat_deg = float(row[key])
            elif "longitude" in key.lower():
                lon_deg = float(row[key])
            elif "altitude" in key.lower():
                alt_m = float(row[key])
            elif "objectid" in key.lower():
                id = int(row[key])
        print(lat_deg, lon_deg, alt_m)
        point = { "id": id, "comment": "",
                  "lat_deg": lat_deg, "lon_deg": lon_deg, "alt_m": alt_m }
        markers.append( point )

filename = os.path.join(args.project, "ImageAnalysis", "annotations.json")
f = open(filename, 'w')
root = { 'id_prefix': args.csv_file,
         'markers': markers }
json.dump(root, f, indent=4)
f.close()
