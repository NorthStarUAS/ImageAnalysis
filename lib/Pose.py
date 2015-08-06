#!/usr/bin/python

# pose related functions

import fileinput
import math

# from Sentera meta data file
def setAircraftPoses(proj, metafile="", force=True, weight=True):
    f = fileinput.input(metafile)
    for line in f:
        line.strip()
        field = line.split(',')
        name = field[0]
        lat = float(field[1])
        lon = float(field[2])
        msl = float(field[3])
        yaw = float(field[4])
        pitch = float(field[5])
        roll = float(field[6])

        image = proj.findImageByName(name)
        if image != None:
            if force or (math.fabs(image.aircraft_lon) < 0.01 and math.fabs(image.aircraft_lat) < 0.01):
                image.set_aircraft_pose( lon, lat, msl, roll, pitch, yaw )
                image.weight = 1.0
                image.save_meta()
                print "%s roll=%.1f pitch=%.1f weight=%.2f" % (image.name, roll, pitch, image.weight)
