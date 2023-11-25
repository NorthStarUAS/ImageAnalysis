#!/usr/bin/python

# routines to query image meta data

import datetime
import piexif
# from libxmp.utils import file_to_dict # pip install python-xmp-toolkit

from .logger import log

def get_camera_info(image_file):
    camera = ""
    make = ""
    model = ""
    exif_dict = piexif.load(image_file)
    if piexif.ImageIFD.Make in exif_dict['0th']:
        make = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8').rstrip('\x00')
        camera = make
    if piexif.ImageIFD.Model in exif_dict['0th']:
        model = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8').rstrip('\x00')
        camera += '_' + model
    if piexif.ExifIFD.LensModel in exif_dict['Exif']:
        lens_model = exif_dict['Exif'][piexif.ExifIFD.LensModel].decode('utf-8').rstrip('\x00')
        camera += '_' + lens_model
    else:
        lens_model = None
    camera = camera.replace(' ', '_')
    return camera, make, model, lens_model

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

def get_pose(image_file):
    # exif data
    exif_dict = piexif.load(image_file)

    # extended xmp tags (hack)
    fd = open(image_file, "rb")
    d = str(fd.read())
    xmp_start = d.find('<x:xmpmeta')
    xmp_end = d.find('</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end+12]
    #print(xmp_str)
    lines = xmp_str.split("\\n")
    #print(lines)
    xmp = {}
    for line in lines:
        line = line.rstrip().lstrip()
        if line[0] == "<":
            continue
        token, val = line.split("=")
        val = val.strip('"')
        # print(token, val)
        xmp[token] = val

    #for key in xmp:
    #    print(key, xmp[key])

    # for ifd in exif_dict:
    #     if ifd == str("thumbnail"):
    #         print("thumb thumb thumbnail")
    #         continue
    #     print(ifd, ":")
    #     for tag in exif_dict[ifd]:
    #         print(ifd, tag, piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

    if 'drone-dji:GpsLatitude' in xmp:
        lat_deg = float(xmp['drone-dji:GpsLatitude'])
    else:
        elat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
        lat_deg = dms_to_decimal(elat[0], elat[1], elat[2],
                                 exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode('utf-8'))

    if 'drone-dji:GpsLongitude' in xmp:
        lon_deg = float(xmp['drone-dji:GpsLongitude'])
    else:
        elon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
        lon_deg = dms_to_decimal(elon[0], elon[1], elon[2],
                                 exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode('utf-8'))

    if 'drone-dji:AbsoluteAltitude' in xmp:
        alt_m = float(xmp['drone-dji:AbsoluteAltitude'])
        if alt_m < 0:
            log("image meta data is reporting negative absolute alitude!")
    else:
        ealt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
        alt_m = ealt[0] / ealt[1]

    #exif_dict[GPS + 'MapDatum'])
    #print('lon ref', exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef])

    # print exif.exif_keys
    if piexif.ImageIFD.DateTime in exif_dict['0th']:
        strdate, strtime = exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8').split()
        year, month, day = strdate.split(':')
        hour, minute, second = strtime.split(':')
        d = datetime.date(int(year), int(month), int(day))
        t = datetime.time(int(hour), int(minute), int(second))
        dt = datetime.datetime.combine(d, t)
        unixtime = float(dt.strftime('%s'))
    else:
        unixtime = None
    #print('pos:', lat, lon, alt, heading)

    if "tiff:Model" in xmp and xmp["tiff:Model"] == "FC7303" and 'drone-dji:FlightYawDegree' in xmp:
        # mavic mini 2 doesn't report gimbal yaw, just flight yaw
        yaw_deg = float(xmp['drone-dji:FlightYawDegree'])
        while yaw_deg < 0:
            yaw_deg += 360
    elif 'drone-dji:GimbalYawDegree' in xmp:
        yaw_deg = float(xmp['drone-dji:GimbalYawDegree'])
        while yaw_deg < 0:
            yaw_deg += 360
    elif 'Camera:Yaw' in xmp:
        yaw_deg = float(xmp['Camera:Yaw'])
        while yaw_deg < 0:
            yaw_deg += 360
    else:
        yaw_deg = None
    print("yaw_deg:", yaw_deg)

    if 'drone-dji:GimbalPitchDegree' in xmp:
        pitch_deg = float(xmp['drone-dji:GimbalPitchDegree'])
    elif 'Camera:Pitch' in xmp:
        pitch_deg = float(xmp['Camera:Pitch'])
    else:
        pitch_deg = None

    if 'drone-dji:GimbalRollDegree' in xmp:
        roll_deg = float(xmp['drone-dji:GimbalRollDegree'])
    elif 'Camera:Roll' in xmp:
        roll_deg = float(xmp['Camera:Roll'])
    else:
        roll_deg = None

    return lon_deg, lat_deg, alt_m, unixtime, yaw_deg, pitch_deg, roll_deg
