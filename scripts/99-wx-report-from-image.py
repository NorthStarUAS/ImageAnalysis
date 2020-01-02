#!/usr/bin/python3

# Weather Summary

import argparse
import datetime
import os
import piexif
from libxmp.utils import file_to_dict

from lib import srtm

parser = argparse.ArgumentParser(description="Lookup a weather report for the location/time an image was captured.")
parser.add_argument("--image", required=False, help="geotagged image")
parser.add_argument("--lat", required=False, type=float, help="image lat")
parser.add_argument("--lon", required=False, type=float, help="image lon")
parser.add_argument("--unixtime", required=False, type=int, help="unix sec")
args = parser.parse_args()

def dms_to_decimal(degrees, minutes, seconds, sign=' '):
    """Convert degrees, minutes, seconds into decimal degrees.

    >>> dms_to_decimal(10, 10, 10)
    10.169444444444444
    >>> dms_to_decimal(8, 9, 10, 'S')
    -8.152777777777779
    """
    return (-1 if sign[0] in 'SWsw' else 1) * (
        float(degrees[0] / degrees[1])        +
        float(minutes[0] / minutes[1]) / 60   +
        float(seconds[0] / seconds[1]) / 3600
    )

from os.path import expanduser
home = expanduser("~")
keyfile = os.path.join(home, '.forecastio')
if not os.path.isfile(keyfile):
    print("you must sign up for a free apikey at forecast.io and insert it as a single line inside a file called ~/.forecastio (with no other text in the file)")
    quit()
    
fio = open(home + '/.forecastio')
apikey = fio.read().rstrip()
fio.close()
if not len(apikey):
    print("Cannot lookup weather because no forecastio apikey found.")
    quit()
    
alt = 0
if args.image:
    exif_dict = piexif.load(args.image)
    if False:
        for ifd in exif_dict:
            if ifd == str("thumbnail"):
                print("thumb thumb thumbnail")
                continue
            print(ifd, ":")
            for tag in exif_dict[ifd]:
                print(ifd, tag, piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

    elat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
    lat = dms_to_decimal(elat[0], elat[1], elat[2],
                         exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode('utf-8'))
    elon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
    lon = dms_to_decimal(elon[0], elon[1], elon[2],
                         exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode('utf-8'))
    #print(lon)
    ealt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
    alt = ealt[0] / ealt[1]
    #exif_dict[GPS + 'MapDatum'])
    #print('lon ref', exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef])

    # print exif.exif_keys
    if piexif.ImageIFD.DateTime in exif_dict['0th']:
        strdate, strtime = exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8').split()
        print(strdate, strtime)
        year, month, day = strdate.split(':')
        hour, minute, second = strtime.split(':')
        #d = datetime.date(int(year), int(month), int(day))
        #t = datetime.time(int(hour), int(minute), int(second))
        #dt = datetime.datetime.combine(d, t) 
        dt = datetime.datetime(int(year), int(month), int(day),
                               int(hour), int(minute), int(second))
        unix_sec = float(dt.strftime('%s'))

else:
    lat = args.lat
    lon = args.lon
    unix_sec = args.unixtime

print('pos, time:', lat, lon, alt, unix_sec)
    
if unix_sec < 1:
    print("Cannot lookup weather because gps didn't report unix time.")
else:
    print("## Weather")
    d = datetime.datetime.utcfromtimestamp(unix_sec)
    print(d.strftime("%Y-%m-%d-%H:%M:%S"))

    url = 'https://api.darksky.net/forecast/' + apikey + '/%.8f,%.8f,%.d' % (lat, lon, unix_sec)

    import urllib.request, json
    response = urllib.request.urlopen(url)
    wx = json.loads(response.read())
    mph2kt = 0.868976
    mb2inhg = 0.0295299830714
    if 'currently' in wx:
        currently = wx['currently']
        #for key in currently:
        #    print key, ':', currently[key]
        if 'icon' in currently:
            icon = currently['icon']
            print("- Conditions: " + icon)
        if 'temperature' in currently:
            tempF = currently['temperature']
            tempC = (tempF - 32.0) * 5 / 9
            print("- Temperature: %.1f F" % tempF + " (%.1f C)" % tempC)
        if 'dewPoint' in currently:
            dewF = currently['dewPoint']
            dewC = (dewF - 32.0) * 5 / 9
            print("- Dewpoint: %.1f F" % dewF + " (%.1f C)" % dewC)
        if 'humidity' in currently:
            hum = currently['humidity']
            print("- Humidity: %.0f%%" % (hum * 100.0))
        if 'pressure' in currently:
            mbar = currently['pressure']
            inhg = mbar * mb2inhg
        else:
            mbar = 0
            inhg = 11.11
        print("- Pressure: %.2f inhg" % inhg + " (%.1f mbar)" % mbar)
        if 'windSpeed' in currently:
            wind_mph = currently['windSpeed']
            wind_kts = wind_mph * mph2kt
        else:
            wind_mph = 0
            wind_kts = 0
        if 'windBearing' in currently:
            wind_deg = currently['windBearing']
        else:
            wind_deg = 0
        print("- Wind %d deg @ %.1f kt (%.1f mph)" % (wind_deg, wind_kts, wind_mph) + "\n")
        if 'visibility' in currently:
            vis = currently['visibility']
            print("- Visibility: %.1f miles" % vis)
        if 'cloudCover' in currently:
            cov = currently['cloudCover']
            print("- Cloud Cover: %.0f%%" % (cov * 100.0))
        print("- METAR: KXYZ " + d.strftime("%d%H%M") + "Z" +
                " %03d%02dKT" % (round(wind_deg/10)*10, wind_kts) +
                " " + ("%.1f" % vis).rstrip('0').rstrip(".") + "SM" +
                " " + ("%.0f" % tempC).replace('-', 'M') + "/" +
                ("%.0f" % dewC).replace('-', 'M') +
                " A%.0f=\n" % (inhg*100)
        )

ref = [ lat, lon, 0.0 ]
print("NED reference location:", ref)

# local surface approximation
srtm.initialize( ref, 6000, 6000, 30)
surface = srtm.ned_interp([0, 0])
print("SRTM surface elevation below image: %.1f m (egm96)" % surface)
