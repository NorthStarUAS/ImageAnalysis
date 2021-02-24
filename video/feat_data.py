from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate  # strait up linear interpolation, nothing fancy
import scipy.signal as signal

class FeatureData():
    data = None
    hz = None
    tmin = None
    tmax = None
    span_sec = None
    interp_p = None
    interp_q = None
    interp_r = None

    def __init__(self):
        pass
    
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

    def smooth(self, smooth_cutoff_hz):
        b, a = signal.butter(2, smooth_cutoff_hz, fs=self.hz)
        self.data['p (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['p (rad/sec)'])
        self.data['q (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['q (rad/sec)'])
        self.data['r (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['r (rad/sec)'])
        self.data['hp (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['hp (rad/sec)'])
        self.data['hq (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['hq (rad/sec)'])
        self.data['hr (rad/sec)'] = \
            signal.filtfilt(b, a, self.data['hr (rad/sec)'])

    def make_interp(self):
        self.interp_p = interpolate.interp1d(self.data['video time'],
                                             self.data['hp (rad/sec)'],
                                             bounds_error=False, fill_value=0.0)
        self.interp_q = interpolate.interp1d(self.data['video time'],
                                             self.data['hq (rad/sec)'],
                                             bounds_error=False, fill_value=0.0)
        self.interp_r = interpolate.interp1d(self.data['video time'],
                                             self.data['hr (rad/sec)'],
                                             bounds_error=False, fill_value=0.0)
        
    def get_vals(self, x):
        return self.interp_p(x), self.interp_q(x), self.interp_r(x)
    
    def resample(self, sample_hz):
        result = []
        print("video range = %.3f - %.3f (%.3f)" % (self.tmin, self.tmax, self.tmax-self.tmin))
        for x in np.linspace(self.tmin, self.tmax, int(round(self.span_sec*sample_hz))):
            p, q, r = self.get_vals(x)
            result.append( [x, p, q, r] )
        print("video data len:", len(result))
        return result

    def plot(self):
        plt.figure()
        plt.plot(self.data['p (rad/sec)'], label="p")
        plt.plot(self.data['q (rad/sec)'], label="q")
        plt.plot(self.data['r (rad/sec)'], label="r")
        plt.xlabel("Video time (sec)")
        plt.ylabel("rad/sec")
        plt.legend()
        plt.show()
