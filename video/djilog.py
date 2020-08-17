# module to parse and return information from a "verbose" dji flight
# log exported as csv

# note, the flight must be conducted with the dji go app (which makes
# sense if the collected data is a video.)  You must upload the flight
# log from your ipad (android?) to this web site:
# https://www.phantomhelp.com/logviewer/upload/ and then download the
# "verbose" csv file, extract the zip and tada.

import csv
import numpy as np
from scipy import interpolate

import time
import datetime
import calendar
import pytz

class djicsv:
    
    def __init__(self):
        self.log = []

    def load(self, file_name):
        with open(file_name, "r", encoding='ISO-8859-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                #print(row)
                time_str = row['CUSTOM.updateTime']
                print(time_str)
                if len(time_str.split(".")) == 1:
                    date_time_obj = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
                else:
                    date_time_obj = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S.%f')
                unix_sec = (date_time_obj - datetime.datetime(1970, 1, 1)).total_seconds()
                print(unix_sec)
                #strdate, strtime = time_str.split()
                #year, month, day = strdate.split('/')
                #hour, minute, second_str = strtime.split(':')
                #second = int(float(second_str))
                #sec_frac = float(second_str) - second
                #dt = datetime.datetime( int(year), int(month), int(day),
                #                        int(hour), int(minute), int(second),
                #                        tzinfo=pytz.UTC )
                #unixtime = float(dt.strftime('%s')) + sec_frac
                #print(strdate, ":", strtime, unixtime)
                #print(date_time_obj.strftime('%s'))
                record = {
                    'time_str': row['CUSTOM.updateTime'],
                    'unix_sec': unix_sec,
                    'lat': row['OSD.latitude'],
                    'lon': row['OSD.longitude'],
                    'baro_alt': row['OSD.altitude [m]'],
                    'pitch': row['GIMBAL.pitch'],
                    'roll': row['GIMBAL.roll'],
                    'yaw': row['GIMBAL.yaw']
                }
                self.log.append(record)

        # setup interpolation structures
        columns = {}
        for key in self.log[0]:
            if type(self.log[0][key]) != str:
                columns[key] = []
        for record in self.log:
            for key in columns:
                columns[key].append(record[key])
        self.interp = {}
        for key in columns:
            self.interp[key] = interpolate.interp1d(columns['unix_sec'],
                                                    columns[key],
                                                    bounds_error=False,
                                                    fill_value=0.0)

    def query(self, t):
        result = {}
        result['unix_time'] = t
        for key in self.interp:
            result[key] = self.interp[key](t).item()
        return result
