# module to parse and return information from a "verbose" dji flight
# log exported as csv

# note, the flight must be conducted with the dji go app (which makes
# sense if the collected data is a video.)  You must upload the flight
# log from your ipad (android?) to this web site:
# https://www.phantomhelp.com/logviewer/upload/ and then download the
# "verbose" csv file, extract the zip and tada.

import csv
import numpy as np
import re
from scipy import interpolate

import datetime

class djicsv:
    def __init__(self):
        self.log = []

    def load(self, file_name):
        # DJIFlightRecord_2021-07-29_\[16-32-50\].csv
        pos = file_name.find("DJIFlightRecord_")
        year = file_name[pos+16:pos+20]
        month = file_name[pos+21:pos+23]
        day = file_name[pos+24:pos+26]
        print(year, month, day)
        #with open(file_name, "r", encoding='ISO-8859-1') as f:
        with open(file_name, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(row)
                time_str = row['CUSTOM.updateTime [local]']
                (t, ampm) = time_str.split(" ")
                print(t, ampm)
                (hour, min, sec) = t.split(":")
                hour = int(hour)
                if ampm == "PM":
                    hour += 12
                print(hour, min, sec)
                time_str = "%s/%s/%s %02d:%s:%s" % (year, month, day,
                                                     hour, min, sec)
                print(time_str)
                if len(time_str.split(".")) == 1:
                    date_time_obj = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
                else:
                    date_time_obj = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S.%f')
                unix_sec = (date_time_obj - datetime.datetime(1970, 1, 1)).total_seconds()
                #print(unix_sec)
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
                    'time_str': time_str,
                    'unix_sec': unix_sec,
                    'lat': float(row[' OSD.latitude']),
                    'lon': float(row[' OSD.longitude']),
                    'baro_alt': float(row[' OSD.altitude [ft]']),
                    'pitch': float(row[' GIMBAL.pitch']),
                    'roll': float(row[' GIMBAL.roll']),
                    'yaw': float(row[' GIMBAL.yaw'])
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

class djisrt:
    def __init__(self):
        self.need_interpolate = False
        self.times = []
        self.lats = []
        self.lons = []
        self.heights = []

    def load(self, srt_name):
        # read and parse srt file, setup data interpolator
        self.need_interpolate = False
        self.times = []
        self.lats = []
        self.lons = []
        self.heights = []
        ts = 0
        lat = 0
        lon = 0
        height = 0
        with open(srt_name, 'r') as f:
            state = 0
            for line in f:
                if line.rstrip() == "":
                    state = 0
                elif state == 0:
                    counter = int(line.rstrip())
                    state += 1
                    # print(counter)
                elif state == 1:
                    time_range = line.rstrip()
                    (start, end) = time_range.split(' --> ')
                    (shr, smin, ssec_txt) = start.split(':')
                    (ssec, ssubsec) = ssec_txt.split(',')
                    (ehr, emin, esec_txt) = end.split(':')
                    (esec, esubsec) = esec_txt.split(',')
                    ts = int(shr)*3600 + int(smin)*60 + int(ssec) + int(ssubsec)/1000
                    te = int(ehr)*3600 + int(emin)*60 + int(esec) + int(esubsec)/1000
                    print(ts, te)
                    state += 1
                elif state == 2:
                    # check for phantom (old) versus mavic2 (new) record
                    data_line = line.rstrip()
                    if re.search('\<font.*\>', data_line):
                        # mavic 2
                        state += 1
                    else:
                        # phantom
                        self.need_interpolate = True
                        m = re.search('(?<=GPS \()(.+)\)', data_line)
                        (lon_txt, lat_txt, alt) = m.group(0).split(', ')
                        m = re.search('(?<=, H )([\d\.]+)', data_line)
                        if lat_txt != 'n/a':
                            lat = float(lat_txt)
                        if lon_txt != 'n/a':
                            lon = float(lon_txt)
                        height = float(m.group(0))
                        # print('gps:', lat, lon, height)
                        self.times.append(ts)
                        self.lats.append(lat)
                        self.lons.append(lon)
                        self.heights.append(height)
                        state = 0
                elif state == 3:
                    # mavic 2 datetime line
                    datetime = line.rstrip()
                    state += 1
                elif state == 4:
                    # mavic 2 big data line
                    data_line = line.rstrip()
                    m = re.search('latitude : ([+-]?\d*\.\d*)', data_line)
                    if m:
                        lat = float(m.group(1))
                    else:
                        lat = None
                    m = re.search('longt?itude : ([+-]?\d*\.\d*)', data_line)
                    if m:
                        lon = float(m.group(1))
                    else:
                        lon = None
                    m = re.search('altitude.*: ([+-]?\d*\.\d*)', data_line)
                    if m:
                        alt = float(m.group(1))
                    else:
                        alt = None
                    self.times.append(datetime)
                    self.lats.append(lat)
                    self.lons.append(lon)
                    self.heights.append(alt)
                else:
                    pass

        if self.need_interpolate:
            print('setting up interpolators')
            self.interp_lats = \
                interpolate.interp1d(self.times, self.lats,
                                     bounds_error=False, fill_value=0.0)
            self.interp_lons = \
                interpolate.interp1d(self.times, self.lons,
                                     bounds_error=False, fill_value=0.0)
            self.interp_heights = \
                interpolate.interp1d(self.times, self.heights,
                                     bounds_error=False, fill_value=0.0)

