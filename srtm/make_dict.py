#!/usr/bin/python

import fileinput
import json

url_base = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3'

regions = ['Africa', 'Australia', 'Eurasia', 'Islands', 'North_America',
           'South_America']

srtm_dict = {}

srtm_directory = 'srtm.json'

for region in regions:
    print 'Processing', region
    f = fileinput.input(region)
    for name in f:
        name = name.strip()
        url = url_base + '/' + region + '/' + name
        key = name.replace('.hgt.zip', '')
        srtm_dict[key] = url

try:
    print "Writing", srtm_directory
    f = open(srtm_directory, 'w')
    json.dump(srtm_dict, f, indent=2, sort_keys=True)
    f.close()
except IOError as e:
    print "Save srtm_dict(): I/O error({0}): {1}".format(e.errno, e.strerror)
  
