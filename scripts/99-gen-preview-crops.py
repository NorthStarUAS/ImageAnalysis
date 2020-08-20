#!/usr/bin/python3

import argparse
import cv2
import fnmatch
import json
import numpy as np
import os

import navpy
from props import getNode       # aura-props

from lib import camera
from lib import project

parser = argparse.ArgumentParser(description='Generate cropped preview images from annotation points.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

id_prefix = "Marker "

file = os.path.join(args.project, 'ImageAnalysis', 'annotations.json')
if os.path.exists(file):
    print('Loading annotations:', file)
    f = open(file, 'r')
    root = json.load(f)
    if type(root) is dict:
        if 'id_prefix' in root:
            id_prefix = root['id_prefix']
        if 'markers' in root:
            lla_list = root['markers']
    f.close()
else:
    print('No annotations file found.')
    quit()
    
proj = project.ProjectMgr(args.project)
proj.load_images_info()

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ned_ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]
print(ned_ref)

K = camera.get_K()
dist_coeffs = camera.get_dist_coeffs()

# make sure annotation preview directory exists and is empty:
preview_dir = os.path.join(args.project, "ImageAnalysis", "annotations-preview")
if not os.path.exists(preview_dir):
    print("Making previews directory:", preview_dir)
    os.makedirs(preview_dir)
else:
    # clean any past previews out
    for file in os.listdir(preview_dir):
        if fnmatch.fnmatch(file, '*.jpg'):
            preview_file = os.path.join(preview_dir, file)
            print("Removing old preview file:", preview_file)
            os.remove(preview_file)

# leaflet header
html_file = os.path.join(preview_dir, "newindex.html")
f = open(html_file, 'w')
f.write(
"""
<!DOCTYPE html>
<html>
  <head>
	
    <title>Quick Start - Leaflet</title>

    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	
    <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.6.0/dist/leaflet.css" integrity="sha512-xwE/Az9zrjBIphAcBb3F6JVqxf46+CDLwfLMHloNu6KEQCAWi6HcDUbeOfBIptF7tcCzusKFjFw2yuvEpDL9wQ==" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js" integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew==" crossorigin=""></script>
    <script src="leaflet-bing-layer.js"></script>

  </head>
"""
)

# leaflet body
f.write(
"""
  <body>

    <div id="mapid" style="width:100%; height:800px;"></div>

    <script>

    var mymap = L.map('mapid');

    // API key for bing. Please get your own at: http://bingmapsportal.com/ 
    var apiKey = "AmT3B1o5RmNfyBsZ634rbefWuNbsHJsgTcyGWILtBrU74iDpQwikazUVu9TT8ZTL";

    var baselayer = {
        "OpenStreetMap": new L.TileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            minZoom: 0,
            maxZoom: 18,
            attribution: 'Map data &copy; <a target="_blank" href="http://openstreetmap.org">OpenStreetMap</a> contributors'
        }),
        "Carto Light": new L.TileLayer('http://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
            minZoom: 0,
            maxZoom: 18,
            attribution: 'Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
        }),
        "Carto Dark": new L.TileLayer('http://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', {
            minZoom: 0,
            maxZoom: 18,
            attribution: 'Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
        }),
        "Bing": L.tileLayer.bing({
            bingMapsKey: apiKey,
            imagerySet: 'AerialWithLabels',
            attribution: 'bing'
        }),
    }

    mymap.addLayer(baselayer["OpenStreetMap"]);
    L.control.layers(baselayer).addTo(mymap);

"""
)

min_lat = None
max_lat = None
min_lon = None
max_lon = None

for m in lla_list:
    feat_ned = navpy.lla2ned(m['lat_deg'], m['lon_deg'], m['alt_m'],
                             ned_ref[0], ned_ref[1], ned_ref[2])
    print("Feature id:", m['id'], "pos:", m['lat_deg'], m['lon_deg'], m['alt_m'])

    if min_lat is None:
        min_lat = m['lat_deg']
        max_lat = m['lat_deg']
        min_lon = m['lon_deg']
        max_lon = m['lon_deg']
    else:
        if m['lat_deg'] < min_lat: min_lat = m['lat_deg']
        if m['lat_deg'] > max_lat: max_lat = m['lat_deg']
        if m['lon_deg'] < min_lon: min_lon = m['lon_deg']
        if m['lon_deg'] > max_lon: max_lon = m['lon_deg']
        
    # quick hack to find closest image
    best_dist = None
    best_image = None
    for i in proj.image_list:
        image_ned, ypr, quat = i.get_camera_pose(opt=True)
        dist = np.linalg.norm( np.array(feat_ned) - np.array(image_ned) )
        if best_dist == None or dist < best_dist:
            best_dist = dist
            best_image = i
            # print("  best_dist:", best_dist)
    if best_image != None:
        # project the feature ned coordinate into the uv space of the
        # closest image
        print("  best image:", best_image.name, best_dist)
        rvec, tvec = best_image.get_proj()
        reproj_points, jac = cv2.projectPoints(np.array([feat_ned]),
                                               rvec, tvec,
                                               K, dist_coeffs)
        reproj_list = reproj_points.reshape(-1,2).tolist()
        kp = reproj_list[0]
        print("  estimate pixel location:", kp)
        
        rgb = best_image.load_rgb()
        h, w = rgb.shape[:2]
        size = 256              # actual size is double this
        size2 = size * 2
        cx = int(round(kp[0]))
        cy = int(round(kp[1]))
        if cx < size:
            xshift = size - cx
            cx = size
        elif cx > (w - size):
            xshift = (w - size) - cx
            cx = w - size
        else:
            xshift = 0
        if cy < size:
            yshift = size - cy
            cy = size
        elif cy > (h - size):
            yshift = (h - size) - cy
            cy = h - size
        else:
            yshift = 0
        #print('size:', w, h, 'shift:', xshift, yshift)
        crop = rgb[cy-size:cy+size, cx-size:cx+size]
        label = "%s%03d" % (id_prefix, m['id'])
        preview_file = os.path.join(preview_dir, label + ".jpg")
        print("  Writing preview file:", label + ".jpg")
        cv2.imwrite(preview_file, crop)
        #cv2.imshow(best_image.name + ' ' + label, crop)
        #cv2.waitKey()

        f.write('L.marker([%.10f, %.10f]).addTo(mymap).bindPopup("<img width=\\"%d\\" height=\\"%d\\" src=\\"%s\\"/><p>Hello world!</p><br />I am a popup.", { maxWidth: %d} );\n' %
        ( m['lat_deg'], m['lon_deg'], size2, size2, label + ".jpg", size2)
                )
        
f.write("    mymap.fitBounds([[%.10f, %.10f], [%.10f,%.10f]]);\n" %
        (min_lat, min_lon, max_lat, max_lon))

# leaflet footer
f.write(
"""
  </script>


  </body>
</html>
"""
)
