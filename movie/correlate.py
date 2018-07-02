import csv
import math
from matplotlib import pyplot as plt 
import numpy as np
from scipy import interpolate # strait up linear interpolation, nothing fancy

r2d = 180.0 / math.pi

def sync_clocks(data, interp, movie_log, hz=60, force_shift=None, plot=True):
    x = interp.imu_time
    flight_min = x.min()
    flight_max = x.max()
    print("flight range = %.3f - %.3f (%.3f)" % (flight_min, flight_max, flight_max-flight_min))

    # load movie log
    movie = []
    with open(movie_log, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = [ float(row['frame']), float(row['time']),
                       float(row['rotation (deg)']),
                       float(row['translation x (px)']),
                       float(row['translation y (px)']) ]
            movie.append( record )

    # set approximate camera orienation (front, down, and rear supported)
    cam_facing = 'front'

    # resample movie data
    movie = np.array(movie, dtype=float)
    movie_interp = []
    x = movie[:,1]
    movie_spl_roll = interpolate.interp1d(x, movie[:,2], bounds_error=False, fill_value=0.0)
    movie_spl_pitch = interpolate.interp1d(x, movie[:,3], bounds_error=False, fill_value=0.0)
    movie_spl_yaw = interpolate.interp1d(x, movie[:,4], bounds_error=False, fill_value=0.0)
    xmin = x.min()
    xmax = x.max()
    print("movie range = %.3f - %.3f (%.3f)" % (xmin, xmax, xmax-xmin))
    movie_len = xmax - xmin
    for x in np.linspace(xmin, xmax, movie_len*hz):
        if cam_facing == 'front' or cam_facing == 'down':
            movie_interp.append( [x, movie_spl_roll(x)] )
            #movie_interp.append( [x, -movie_spl_yaw(x)] ) # test, fixme
        else:
            movie_interp.append( [x, -movie_spl_roll(x)] )
            print("movie len:", len(movie_interp))

    # resample flight data
    flight_interp = []
    if cam_facing == 'front' or cam_facing == 'rear':
        y_spline = interp.imu_p     # front/rear facing camera
        #y_spline = interp.imu_r     # front/rear facing camera, test fixme
    else:
        y_spline = interp.imu_r     # down facing camera

    time = flight_max - flight_min
    for x in np.linspace(flight_min, flight_max, time*hz):
        flight_interp.append( [x, y_spline(x)] )
        #print "flight len:", len(flight_interp)

    # compute best correlation between movie and flight data logs
    movie_interp = np.array(movie_interp, dtype=float)
    flight_interp = np.array(flight_interp, dtype=float)

    do_butter_smooth = True
    if do_butter_smooth:
        # maybe filtering video estimate helps something?
        import scipy.signal as signal
        b, a = signal.butter(2, 10.0/(200.0/2))
        flight_butter = signal.filtfilt(b, a, flight_interp[:,1])
        movie_butter = signal.filtfilt(b, a, movie_interp[:,1])
        ycorr = np.correlate(flight_butter, movie_butter, mode='full')
    else:
        ycorr = np.correlate(flight_interp[:,1], movie_interp[:,1], mode='full')

    # display some stats/info
    max_index = np.argmax(ycorr)
    print("max index:", max_index)

    # shift = np.argmax(ycorr) - len(flight_interp)
    # print "shift (pos):", shift
    # start_diff = flight_interp[0][0] - movie_interp[0][0]
    # print "start time diff:", start_diff
    # time_shift = start_diff - (shift/hz)
    # print "movie time shift:", time_shift

    # need to subtract movie_len off peak point time because of how
    # correlate works and shifts against every possible overlap
    shift_sec = np.argmax(ycorr) / hz - movie_len
    print("shift (sec):", shift_sec)
    print(flight_interp[0][0], movie_interp[0][0])
    start_diff = flight_interp[0][0] - movie_interp[0][0]
    print("start time diff:", start_diff)
    time_shift = start_diff + shift_sec

    # estimate  tx, ty vs. r, q multiplier
    tmin = np.amax( [np.amin(movie_interp[:,0]) + time_shift,
                    np.amin(flight_interp[:,0]) ] )
    tmax = np.amin( [np.amax(movie_interp[:,0]) + time_shift,
                    np.amax(flight_interp[:,0]) ] )
    print("overlap range (flight sec):", tmin, " - ", tmax)

    mqsum = 0.0
    fqsum = 0.0
    mrsum = 0.0
    frsum = 0.0
    count = 0
    qratio = 1.0
    for x in np.linspace(tmin, tmax, (tmax-tmin)*hz):
        mqsum += abs(movie_spl_pitch(x-time_shift))
        mrsum += abs(movie_spl_yaw(x-time_shift))
        fqsum += abs(interp.imu_q(x))
        frsum += abs(interp.imu_r(x))
    if fqsum > 0.001:
        qratio = mqsum / fqsum
    if mrsum > 0.001:
        rratio = -mrsum / frsum
    print("pitch ratio:", qratio)
    print("yaw ratio:", rratio)

    print("correlated time shift:", time_shift)
    if force_shift:
        time_shift = force_shift
        print("time shift override (provided on command line):", time_shift)

    if plot:
        # reformat the data
        flight_imu = []
        for imu in data['imu']:
            flight_imu.append([ imu.time, imu.p, imu.q, imu.r ])
        flight_imu = np.array(flight_imu)

        # plot the data ...
        plt.figure(1)
        plt.ylabel('roll rate (deg per sec)')
        plt.xlabel('flight time (sec)')
        if do_butter_smooth:
            plt.plot(flight_interp[:,0], flight_butter*r2d, label='flight data log')
            plt.plot(movie_interp[:,0] + time_shift, movie_butter*r2d, label='smoothed estimate from flight movie')
        else:
            plt.plot(movie[:,1] + time_shift, movie[:,2]*r2d, label='estimate from flight movie')
            # down facing:
            # plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
            # front facing:
            plt.plot(flight_imu[:,0], flight_imu[:,1]*r2d, label='flight data log')
        plt.legend()

        plt.figure(2)
        plt.plot(ycorr)

        plt.figure(3)
        plt.ylabel('pitch rate (deg per sec)')
        plt.xlabel('flight time (sec)')
        plt.plot(movie[:,1] + time_shift, (movie[:,3]/qratio)*r2d, label='estimate from flight movie')
        plt.plot(flight_imu[:,0], flight_imu[:,2]*r2d, label='flight data log')
        plt.legend()

        plt.figure(4)
        plt.ylabel('yaw rate (deg per sec)')
        plt.xlabel('flight time (sec)')
        plt.plot(movie[:,1] + time_shift, (movie[:,4]/rratio)*r2d, label='estimate from flight movie')
        plt.plot(flight_imu[:,0], flight_imu[:,3]*r2d, label='flight data log')
        plt.legend()

        plt.show()

    return time_shift, flight_min, flight_max
