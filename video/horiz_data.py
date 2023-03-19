from math import pi
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate  # strait up linear interpolation, nothing fancy
import scipy.signal as signal

# convenience
d2r = pi / 180.0

class HorizonData():
    data = None
    hz = None
    tmin = None
    tmax = None
    span_sec = None
    interp_phi = None
    interp_the = None
    interp_p = None
    interp_q = None

    def __init__(self):
        pass

    # estimate roll/pitch rates from horizon data
    def make_rates(self):
        t_last = -1
        phi_last = 0
        the_last = 0
        p_list = []
        q_list = []
        for i in range(len(self.data['video time'])):
            t = self.data['video time'].iat[i]
            phi = self.data['camera roll (deg)'].iat[i]
            the = self.data['camera pitch (deg)'].iat[i]
            if t_last >= 0:
                delta_phi = phi - phi_last
                delta_the = the - the_last
                delta_t = t - t_last
            else:
                delta_phi = 0
                delta_the = 0
                delta_t = 1
            p = delta_phi * d2r / delta_t
            q = delta_the * d2r / delta_t
            t_last = t
            phi_last = phi
            the_last = the
            p_list.append(p)
            q_list.append(q)
        self.data['p (rad/sec)'] = p_list
        self.data['q (rad/sec)'] = q_list

    def load(self, feature_file):
        self.data = pd.read_csv(feature_file)
        self.data.set_index('video time', inplace=True, drop=False)
        self.tmin = self.data['video time'].min()
        self.tmax = self.data['video time'].max()
        self.span_sec = self.tmax - self.tmin
        feat_count = len(self.data['video time'])
        print("number of video records:", feat_count)
        self.hz = int(round((feat_count / self.span_sec)))
        print("video fs:", self.hz)
        self.make_rates()       # derived values

    def smooth(self, cutoff_hz):
        b, a = signal.butter(2, cutoff_hz, fs=self.hz)
        self.data['camera roll (deg)'] = \
            signal.filtfilt(b, a, self.data['camera roll (deg)'])
        self.data['camera pitch (deg)'] = \
            signal.filtfilt(b, a, self.data['camera pitch (deg)'])
        self.data['p (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['p (rad/sec)'])
        self.data['q (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['q (rad/sec)'])

    def make_interp(self):
        self.interp_phi = interpolate.interp1d(self.data['video time'],
                                               self.data['camera roll (deg)'] * d2r,
                                               bounds_error=False, fill_value=0.0)
        self.interp_the = interpolate.interp1d(self.data['video time'],
                                               self.data['camera pitch (deg)'] * d2r,
                                               bounds_error=False, fill_value=0.0)
        self.interp_p = interpolate.interp1d(self.data['video time'],
                                             self.data['roll rate (rad/sec)'],
                                             bounds_error=False, fill_value=0.0)
        self.interp_q = interpolate.interp1d(self.data['video time'],
                                             self.data['pitch rate (rad/sec)'],
                                             bounds_error=False, fill_value=0.0)

    def get_vals(self, x):
        return self.interp_phi(x), self.interp_the(x), self.interp_p(x), self.interp_q(x)

    def resample(self, sample_hz):
        result = []
        print("video range = %.3f - %.3f (%.3f)" % (self.tmin, self.tmax, self.tmax-self.tmin))
        for x in np.linspace(self.tmin, self.tmax, int(round(self.span_sec*sample_hz))):
            phi, the, p, q = self.get_vals(x)
            result.append( [x, phi, the, p, q] )
        print("video data len:", len(result))
        return result

    def plot(self):
        plt.figure()
        plt.plot(self.data['p (rad/sec)'], label="p")
        plt.plot(self.data['q (rad/sec)'], label="q")
        plt.xlabel("Video time (sec)")
        plt.ylabel("rad/sec")
        plt.legend()
        plt.show()
