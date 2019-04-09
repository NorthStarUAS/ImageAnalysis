import fileinput
import math
import numpy as np
import re
from scipy import interpolate # strait up linear interpolation, nothing fancy

yaw_interp = None
pitch_interp = None
roll_interp = None
north_interp = None
east_interp = None
down_interp = None

def load(filename):
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
