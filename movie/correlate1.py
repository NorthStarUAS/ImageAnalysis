#!/usr/bin/python

import argparse
import math
from matplotlib import pyplot as plt 
import numpy as np
import re
from scipy.interpolate import InterpolatedUnivariateSpline

parser = argparse.ArgumentParser(description='correlate movie data to flight data.')
parser.add_argument('--movie-log', required=True, help='movie data file')
parser.add_argument('--resample-hz', type=float, default=30.0, help='resample rate (hz)')
parser.add_argument('--apm-log', help='APM tlog converted to csv')
parser.add_argument('--aura-log', help='Aura imu.txt file')
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

flight = []
last_time = -1

if args.apm_log:
    # load APM flight log
    with open(args.apm_log, 'rb') as f:
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
elif args.aura_log:
    # load Aura flight log
    with open(args.aura_log, 'rb') as f:
        for line in f:
            tokens = line.rstrip().split()
            timestamp = float(tokens[0])
            if timestamp > last_time:
                flight.append( [tokens[0], tokens[1], tokens[2], tokens[3]] )
                last_time = timestamp
            else:
                print "ERROR: time went backwards:", timestamp, last_time
else:
    print "No flight log specified, cannot continue."
    quit()
    
# resample movie data
movie = np.array(movie, dtype=float)
movie_interp = []
x = movie[:,0]
movie_spl_roll = InterpolatedUnivariateSpline(x, movie[:,2])
movie_spl_pitch = InterpolatedUnivariateSpline(x, movie[:,3])
movie_spl_yaw = InterpolatedUnivariateSpline(x, movie[:,4])
xmin = x.min()
xmax = x.max()
print "movie range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    movie_interp.append( [x, movie_spl_roll(x)] )
print "movie len:", len(movie_interp)

# resample flight data
flight = np.array(flight, dtype=float)
flight_interp = []
x = flight[:,0]
if args.apm_log:
    print "Fixme: pick correct flight log axes"
    quit()
flight_spl_roll = InterpolatedUnivariateSpline(x, flight[:,1])
flight_spl_pitch = InterpolatedUnivariateSpline(x, flight[:,2])
flight_spl_yaw = InterpolatedUnivariateSpline(x, flight[:,3])
xmin = x.min()
xmax = x.max()
print "flight range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin)
time = xmax - xmin
for x in np.linspace(xmin, xmax, time*args.resample_hz):
    flight_interp.append( [x, flight_spl_roll(x)] )
print "flight len:", len(flight_interp)

# compute best correlation between movie and flight data logs
movie_interp = np.array(movie_interp, dtype=float)
flight_interp = np.array(flight_interp, dtype=float)
ycorr = np.correlate(movie_interp[:,1], flight_interp[:,1], mode='full')

# display some stats/info
max_index = np.argmax(ycorr)
print "max index:", max_index
shift = np.argmax(ycorr) - len(flight_interp)
print "shift (pos):", shift
start_diff = flight_interp[0][0] - movie_interp[0][0]
print "start time diff:", start_diff
time_shift = start_diff - (shift/args.resample_hz)
print "movie time shift:", time_shift

# estimate  tx, ty vs. r, q multiplier
tmin = np.amax( [np.amin(movie_interp[:,0]) + time_shift,
                np.amin(flight_interp[:,0]) ] )
tmax = np.amin( [np.amax(movie_interp[:,0]) + time_shift,
                np.amax(flight_interp[:,0]) ] )
print "overlap range (flight sec):", tmin, " - ", tmax

mqsum = 0.0
fqsum = 0.0
mrsum = 0.0
frsum = 0.0
count = 0
qratio = 1.0
for x in np.linspace(tmin, tmax, time*args.resample_hz):
    mqsum += abs(movie_spl_pitch(x-time_shift))
    mrsum += abs(movie_spl_yaw(x-time_shift))
    fqsum += abs(flight_spl_pitch(x))
    frsum += abs(flight_spl_yaw(x))
if fqsum > 0.001:
    qratio = mqsum / fqsum
if mrsum > 0.001:
    rratio = -mrsum / frsum
print "pitch ratio:", qratio
print "yaw ratio:", rratio

plt.figure(1)
plt.ylabel('roll rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie_interp[:,0] + time_shift, movie_interp[:,1]*r2d, label='estimate from flight movie')
plt.plot(flight_interp[:,0], flight_interp[:,1]*r2d, label='flight data log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.figure(2)
plt.plot(ycorr)

plt.figure(3)
plt.ylabel('pitch rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie[:,0] + time_shift, (movie[:,3]/qratio)*r2d, label='estimate from flight movie')
plt.plot(flight[:,0], flight[:,2]*r2d, label='flight data log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.figure(4)
plt.ylabel('yaw rate (deg per sec)')
plt.xlabel('flight time (sec)')
plt.plot(movie[:,0] + time_shift, (movie[:,4]/rratio)*r2d, label='estimate from flight movie')
plt.plot(flight[:,0], flight[:,3]*r2d, label='flight data log')
#plt.plot(movie_interp[:,1])
#plt.plot(flight_interp[:,1])
plt.legend()

plt.show()
