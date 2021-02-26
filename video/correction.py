import fileinput
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy import interpolate # strait up linear interpolation, nothing fancy
import scipy.signal as signal

yaw_interp = None
pitch_interp = None
roll_interp = None
north_interp = None
east_interp = None
down_interp = None

def load_horiz(filename):
    global roll_interp
    global pitch_interp
    
    data = pd.read_csv(filename)
    data.set_index('flight time (sec)', inplace=True, drop=False)

    # time range / hz
    tmin = data['flight time (sec)'].min()
    tmax = data['flight time (sec)'].max()
    span_sec = tmax - tmin
    feat_count = len(data['flight time (sec)'])
    print("number of video records:", feat_count)
    hz = int(round((feat_count / span_sec)))

    # smooth
    cutoff_hz = 1
    b, a = signal.butter(2, cutoff_hz, fs=hz)
    data['ekf roll error (rad)'] = \
        signal.filtfilt(b, a, data['ekf roll error (rad)'])
    data['ekf pitch error (rad)'] = \
        signal.filtfilt(b, a, data['ekf pitch error (rad)'])

    if False:
        plt.figure()
        plt.plot(data['ekf roll error (rad)'], label="roll error")
        plt.plot(data['ekf pitch error (rad)'], label="pitch error")
        plt.xlabel("Flight time (sec)")
        plt.ylabel("Rad")
        plt.legend()
        plt.show()

    # interpolators
    roll_interp = interpolate.interp1d(data['flight time (sec)'], data['ekf roll error (rad)'], bounds_error=False, fill_value=0.0)
    pitch_interp = interpolate.interp1d(data['flight time (sec)'], data['ekf pitch error (rad)'], bounds_error=False, fill_value=0.0)

    
def load_old(filename):
    global yaw_interp
    global pitch_interp
    global roll_interp
    global north_interp
    global east_interp
    global down_interp
    
    f = fileinput.input(filename)
    table = []
    for line in f:
        tokens = re.split('[,\s]+', line.rstrip())
        time = float(tokens[0])
        yaw_error = float(tokens[1])
        pitch_error = float(tokens[2])
        roll_error = float(tokens[3])
        n_error = float(tokens[4])
        e_error = float(tokens[5])
        d_error = float(tokens[6])
        table.append( [ time,
                        yaw_error, pitch_error, roll_error,
                        n_error, e_error, d_error ] )
        
    array = np.array(table)
    x = array[:,0]
    yaw_interp = interpolate.interp1d(x, array[:,1], bounds_error=False, fill_value=0.0)
    pitch_interp = interpolate.interp1d(x, array[:,2], bounds_error=False, fill_value=0.0)
    roll_interp = interpolate.interp1d(x, array[:,3], bounds_error=False, fill_value=0.0)
    north_interp = interpolate.interp1d(x, array[:,4], bounds_error=False, fill_value=0.0)
    east_interp = interpolate.interp1d(x, array[:,5], bounds_error=False, fill_value=0.0)
    down_interp = interpolate.interp1d(x, array[:,6], bounds_error=False, fill_value=0.0)
