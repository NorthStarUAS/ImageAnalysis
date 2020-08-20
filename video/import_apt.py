#!/usr/bin/env python3

import argparse
import gzip

argparser = argparse.ArgumentParser(description='import apt.dat.gz from FlightGear')
argparser.add_argument('--file', help='fgfs apt.dat.gz file')
args = argparser.parse_args()

ft2m = 0.3048

ident = ''
alt = ''
count = 0
lat_sum = 0
lon_sum = 0

print 'Ident,Lat,Lon,Alt'
with gzip.open(args.file, 'rb') as f:
    for line in f:
        tokens = line.split()
        #print tokens
        if len(tokens) and tokens[0] == '1':
            # start of apt record
            if count > 0:
                # output last record
                print '%s,%.8f,%.8f,%.0f' % (ident, lat_sum / count,
                                             lon_sum / count, alt)
            ident = tokens[4]
            alt = float(tokens[1]) * ft2m
            count = 0
            lat_sum = 0
            lon_sum = 0
        elif len(tokens) and tokens[0] == '100':
            # basic data
            lat_sum += float(tokens[9])
            lon_sum += float(tokens[10])
            lat_sum += float(tokens[18])
            lon_sum += float(tokens[19])
            count += 2
if count > 0:
    # output last record
    print '%s,%.8f,%.8f,%.0f' % (ident, lat_sum / count,
                                 lon_sum / count, alt)
