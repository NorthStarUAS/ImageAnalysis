# build linear interpolaters for the 'standard' flight data.

import math
import numpy as np
from scipy import interpolate # strait up linear interpolation, nothing fancy
import sys

# helpful constants
d2r = math.pi / 180.0

# a tricky way to self reference module globals later in a function
this = sys.modules[__name__]

# interpolators
this.imu_time = None
this.imu_p = None
this.imu_q = None
this.imu_r = None
this.imu_ax = None
this.imu_ay = None
this.imu_az = None

this.gps_lat = None
this.gps_lon = None
this.gps_alt = None
this.gps_unixtime = None

this.filter_lat = None
this.filter_lon = None
this.filter_alt = None
this.filter_vn = None
this.filter_ve = None
this.filter_vd = None
this.filter_phi = None
this.filter_the = None
this.filter_psix = None
this.filter_psiy = None

this.air_speed = None
this.air_true_alt = None
this.air_alpha = None
this.air_beta = None

this.pilot_ail = None
this.pilot_ele = None
this.pilot_thr = None
this.pilot_rud = None
this.pilot_auto = None

this.act_ail = None
this.act_ele = None
this.act_thr = None
this.act_rud = None

this.ap_hdgx = None
this.ap_hdgy = None
this.ap_roll = None
this.ap_alt = None
this.ap_pitch = None
this.ap_speed = None

# build the interpolators
def build(flight_data):
    if 'imu' in flight_data:
        table = []
        for imu in flight_data['imu']:
            table.append([imu.time,
                          imu.p, imu.q, imu.r,
                          imu.ax, imu.ay, imu.az,
                          imu.hx, imu.hy, imu.hz])
        array = np.array(table)
        x = array[:,0]
        this.imu_time = x
        this.imu_p = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                     fill_value=0.0)
        this.imu_q = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                     fill_value=0.0)
        this.imu_r = interpolate.interp1d(x, array[:,3], bounds_error=False,
                                     fill_value=0.0)
        this.imu_ax = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                      fill_value=0.0)
        this.imu_ay = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                      fill_value=0.0)
        this.imu_az = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                      fill_value=0.0)
    if 'gps' in flight_data:
        table = []
        for gps in flight_data['gps']:
            table.append([gps.time,
                          gps.lat, gps.lon, gps.alt,
                          gps.vn, gps.ve, gps.vd,
                          gps.unix_sec, gps.sats])
        array = np.array(table)
        x = array[:,0]
        this.gps_lat = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                       fill_value=0.0)
        this.gps_lon = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                       fill_value=0.0)
        this.gps_alt = interpolate.interp1d(x, array[:,3], bounds_error=False,
                                       fill_value=0.0)
        this.gps_unixtime = interpolate.interp1d(x, array[:,7], bounds_error=False,
                                            fill_value=0.0)
    if 'filter' in flight_data:
        table = []
        for filter in flight_data['filter']:
            psix = math.cos(filter.psi)
            psiy = math.sin(filter.psi)
            table.append([filter.time,
                          filter.lat, filter.lon, filter.alt,
                          filter.vn, filter.ve, filter.vd,
                          filter.phi, filter.the, psix, psiy])
        array = np.array(table)
        x = array[:,0]
        this.filter_lat = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                          fill_value=0.0)
        this.filter_lon = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                          fill_value=0.0)
        this.filter_alt = interpolate.interp1d(x, array[:,3], bounds_error=False,
                                          fill_value=0.0)
        this.filter_vn = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                         fill_value=0.0)
        this.filter_ve = interpolate.interp1d(x, array[:,5], bounds_error=False,
                                         fill_value=0.0)
        this.filter_vd = interpolate.interp1d(x, array[:,6], bounds_error=False,
                                         fill_value=0.0)
        this.filter_phi = interpolate.interp1d(x, array[:,7], bounds_error=False,
                                          fill_value=0.0)
        this.filter_the = interpolate.interp1d(x, array[:,8], bounds_error=False,
                                               fill_value=0.0)
        this.filter_psix = interpolate.interp1d(x, array[:,9],
                                                bounds_error=False,
                                                fill_value=0.0)
        this.filter_psiy = interpolate.interp1d(x, array[:,10],
                                                bounds_error=False,
                                                fill_value=0.0)
    if 'air' in flight_data:
        table = []
        for air in flight_data['air']:
            table.append([air.time, air.airspeed, air.alt_true])
        array = np.array(table)
        x = array[:,0]
        this.air_speed = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                         fill_value=0.0)
        this.air_true_alt = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                            fill_value=0.0)
        #if len(flight_air[0]) >= 13:
        #    flight_air_alpha = interpolate.interp1d(x, flight_air[:,11], bounds_error=False, fill_value=0.0)
        #    flight_air_beta = interpolate.interp1d(x, flight_air[:,12], bounds_error=False, fill_value=0.0)
    if 'pilot' in flight_data:
        table = []
        for pilot in flight_data['pilot']:
            table.append([pilot.time,
                          pilot.aileron,
                          pilot.elevator,
                          pilot.throttle,
                          pilot.rudder,
                          pilot.gear,
                          pilot.flaps,
                          pilot.aux1,
                          pilot.auto_manual])
        array = np.array(table)
        x = array[:,0]
        this.pilot_ail = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                         fill_value=0.0)
        this.pilot_ele = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                         fill_value=0.0)
        this.pilot_thr = interpolate.interp1d(x, array[:,3], bounds_error=False,
                                         fill_value=0.0)
        this.pilot_rud = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                         fill_value=0.0)
        this.pilot_auto = interpolate.interp1d(x, array[:,8], bounds_error=False,
                                          fill_value=0.0)
    if 'act' in flight_data:
        table = []
        for act in flight_data['act']:
            table.append([act.time,
                          act.aileron,
                          act.elevator,
                          act.throttle,
                          act.rudder,
                          act.gear,
                          act.flaps,
                          act.aux1,
                          act.auto_manual])
        array = np.array(table)
        x = array[:,0]
        this.act_ail = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                         fill_value=0.0)
        this.act_ele = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                         fill_value=0.0)
        this.act_thr = interpolate.interp1d(x, array[:,3], bounds_error=False,
                                         fill_value=0.0)
        this.act_rud = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                         fill_value=0.0)
    if 'ap' in flight_data:
        table = []
        for ap in flight_data['ap']:
            hdgx = math.cos(ap.hdg*d2r)
            hdgy = math.sin(ap.hdg*d2r)
            table.append([ap.time,
                          hdgx, hdgy,
                          ap.roll,
                          ap.alt,
                          ap.pitch,
                          ap.speed])
        array = np.array(table)
        x = array[:,0]
        this.ap_hdgx = interpolate.interp1d(x, array[:,1], bounds_error=False,
                                            fill_value=0.0)
        this.ap_hdgy = interpolate.interp1d(x, array[:,2], bounds_error=False,
                                            fill_value=0.0)
        this.ap_roll = interpolate.interp1d(x, array[:,3], bounds_error=False,
                                            fill_value=0.0)
        this.ap_alt = interpolate.interp1d(x, array[:,4], bounds_error=False,
                                           fill_value=0.0)
        this.ap_pitch = interpolate.interp1d(x, array[:,5], bounds_error=False,
                                             fill_value=0.0)
        this.ap_speed = interpolate.interp1d(x, array[:,6], bounds_error=False,
                                             fill_value=0.0)
