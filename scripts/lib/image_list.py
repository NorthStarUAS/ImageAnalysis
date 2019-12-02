# a collection of non-class routines that perform operations on a list
# of images

import math

from . import Image

# return the bounds of the rectangle spanned by the provided list of
# images
def coverage(image_list):
    xmin = None; xmax = None; ymin = None; ymax = None
    for image in image_list:
        (x0, y0, x1, y1) = image.coverage()
        if xmin == None or x0 < xmin:
            xmin = x0
        if ymin == None or y0 < ymin:
            ymin = y0
        if xmax == None or x1 > xmax:
            xmax = x1
        if ymax == None or y1 > ymax:
            ymax = y1
    print("List area coverage: (%.2f %.2f) (%.2f %.2f)" \
        % (xmin, ymin, xmax, ymax))
    return (xmin, ymin, xmax, ymax)

# return True/False if the given rectangles overlap
def rectanglesOverlap(r1, r2):
    (ax0, ay0, ax1, ay1) = r1
    (bx0, by0, bx1, by1) = r2
    if ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0:
        return True
    else:
        return False

# return a list of images that intersect the given rectangle
def getImagesCoveringRectangle(image_list, r2, only_placed=False):
    # build list of images covering target point
    coverage_list = []
    for image in image_list:
        r1 = image.coverage()
        if only_placed and not image.placed:
            continue
        if rectanglesOverlap(r1, r2):
            coverage_list.append(image)
    return coverage_list

# return a list of images that cover the given point within 'pad'
# or are within 'pad' distance of touching the point.
def getImagesCoveringPoint(image_list, x=0.0, y=0.0, pad=20.0, only_placed=False):
    # build list of images covering target point
    coverage_list = []
    bx0 = x-pad
    by0 = y-pad
    bx1 = x+pad
    by1 = y+pad
    r2 = (bx0, by0, bx1, by1)
    coverage_list = getImagesCoveringRectangle(image_list, r2, only_placed)

    name_list = []
    for image in coverage_list:
        name_list.append(image.name)

    print("Images covering point (%.2f %.2f): %s" % (x, y, str(name_list)))

    return coverage_list

def x2lon(self, x):
    nm2m = 1852.0
    x_nm = x / nm2m
    factor = math.cos(self.ref_lat*math.pi/180.0)
    x_deg = (x_nm / 60.0) / factor
    return x_deg + self.ref_lon

def y2lat(self, y):
    nm2m = 1852.0
    y_nm = y / nm2m
    y_deg = y_nm / 60.0
    return y_deg + self.ref_lat

# x, y are in meters ref_lon/lat in degrees
def cart2wgs84( x, y, ref_lon, ref_lat ):
    nm2m = 1852.0
    x_nm = x / nm2m
    y_nm = y / nm2m
    factor = math.cos(ref_lat*math.pi/180.0)
    x_deg = (x_nm / 60.0) / factor + ref_lon
    y_deg = y_nm / 60.0 + ref_lat
    return (x_deg, y_deg)

# x, y are in meters ref_lon/lat in degrees
def wgs842cart( lon_deg, lat_deg, ref_lon, ref_lat ):
    nm2m = 1852.0
    x_deg = lon_deg - ref_lon
    y_deg = lat_deg - ref_lat
    factor = math.cos(ref_lat*math.pi/180.0)
    x_nm = x_deg * 60.0 * factor
    y_nm = y_deg * 60.0
    x_m = x_nm * nm2m
    y_m = y_nm * nm2m
    return (x_m, y_m)
