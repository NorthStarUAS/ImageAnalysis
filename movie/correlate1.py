#!/usr/bin/python

import argparse
import math
from matplotlib import pyplot as plt 
import numpy as np
import re
from scipy.interpolate import InterpolatedUnivariateSpline

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--movie-log', required=True, help='movie data file')
parser.add_argument('--movie-invert-rot', action='store_true', help='(tmp) invert movie log rotation direction.')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--flight-log', required=True, help='APM tlog converted to csv')
args = parser.parse_args()

r2d = 180.0 / math.pi

if args.resample_hz <= 0.001:
    print "Resample rate (hz) needs to be greater than zero."
    quit()
    
# load movie log
movie = []
with open(args.movie_log, 'rb') as f:
    for line in f:
        movie.append( line.rstrip().split() )

# load flight log
flight = []
last_time = -1
with open(args.flight_log, 'rb') as f:
    for line in f:
        if re.search('yawspeed', line):
            tokens = line.rstrip().split(',')
            timestamp = float(tokens[9])/1000.0
            if timestamp > last_time:
                flight.append( [float(tokens[9])/1000.0, float(tokens[17]),
                                float(tokens[19]), float(tokens[21])] )
                last_time = timestamp
            else:
                print "ERROR: time went backwards:", timestamp, last_time
print np.array(flight)

# resample movie data
movie = np.array(movie, dtype=float)
movie_interp = []
x = movie[:,0]
if args.movie_invert_rot:
    y = -movie[:,2]
else:
    y = movie[:,2]
spl = InterpolatedUnivariateSpline(x, y)
xmin = x.min()
xmax = x.max()
print "movie range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    movie_interp.append( [x, spl(x)] )
print "movie len:", len(movie_interp)

# resample flight data
flight = np.array(flight, dtype=float)
flight_interp = []
x = flight[:,0]
y = flight[:,3]
spl = InterpolatedUnivariateSpline(x, y)
xmin = x.min()
xmax = x.max()
print "flight range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    flight_interp.append( [x, spl(x)] )
    #print x, spl(x)
print "flight len:", len(flight_interp)

movie_interp = np.array(movie_interp, dtype=float)
flight_interp = np.array(flight_interp, dtype=float)
ycorr = np.correlate(movie_interp[:,1], flight_interp[:,1], mode='full')
print "ycorr size=", ycorr.size
#for y in ycorr:
#    print y

max_index = np.argmax(ycorr)
print "max index:", max_index
shift = np.argmax(ycorr) - len(flight_interp)
print "shift (pos):", shift
start_diff = flight_interp[0][0] - movie_interp[0][0]
print "start time diff:", start_diff
time_shift = start_diff - (shift/args.resample_hz)
print "movie time shift:", time_shift

plt.figure(1)
plt.ylabel('yaw rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie_interp[:,0] + time_shift, movie_interp[:,1]*r2d, label='estimate from flight movie')
plt.plot(flight_interp[:,0], flight_interp[:,1]*r2d, label='iris flight log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.figure(2)
plt.plot(ycorr)

plt.show()
