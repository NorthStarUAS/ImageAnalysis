# load and query an airport list.

import csv
import math

import navpy

# return a list of airports within range of specified location
def load(file, ned_ref, range_m):
    result = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row['Lat'])
            lon = float(row['Lon'])
            alt = float(row['Alt'])
            pt_ned = navpy.lla2ned( lat, lon, alt,
                                    ned_ref[0], ned_ref[1], ned_ref[2] )
            dist = math.sqrt(pt_ned[0]*pt_ned[0] + pt_ned[1]*pt_ned[1]
                             + pt_ned[2]*pt_ned[2])
            if dist <= range_m:
                print('found:', row['Ident'], 'dist: %.1f km' % (dist/1000))
                result.append( [ row['Ident'], lat, lon, alt ] )
    print('done!')
    return result
            
#load('apt.csv', [45.14, -93.21, 0], 20000)
