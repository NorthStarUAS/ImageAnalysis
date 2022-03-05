"""Pose related functions."""
# pose related functions

import csv
import fileinput
import fnmatch
import math
import os
import re

import navpy
from props import getNode, props_json

from . import camera, exif, image, transformations
from .logger import log

d2r = math.pi / 180.0
r2d = 180.0 / math.pi


def gen_image_list(image_dir):
    """Generate an image list."""
    images = []
    for file in os.listdir(image_dir):
        if fnmatch.fnmatch(file, "*.jpg") or fnmatch.fnmatch(file, "*.JPG"):
            images.append(file)
    return sorted(images)


def set_aircraft_poses(proj, posefile="", order="ypr", max_angle=25.0):
    """Define the image aircraft poses from Sentera meta data file."""
    log("Setting aircraft poses")
    meta_dir = os.path.join(proj.analysis_dir, "meta")
    images_node = getNode("/images", True)

    by_index = False

    f = fileinput.input(posefile)
    for line in f:
        line.strip()
        if re.match(r"^\s*#", line):
            continue
        if re.match(r"^\s*File", line):
            continue
        if re.match(r"^\s*Image", line):
            by_index = True
            file_list = gen_image_list(proj.project_dir)
            continue
        field = line.split(",")
        if not by_index:
            name = field[0]
        else:
            index = int(field[0]) - 1
            name = file_list[index]
        lat_deg = float(field[1])
        lon_deg = float(field[2])
        alt_m = float(field[3])
        if order == "ypr":
            yaw_deg = float(field[4])
            pitch_deg = float(field[5])
            roll_deg = float(field[6])
        elif order == "rpy":
            roll_deg = float(field[4])
            pitch_deg = float(field[5])
            yaw_deg = float(field[6])
        if len(field) >= 8:
            flight_time = float(field[7])
        else:
            flight_time = -1.0

        if not os.path.isfile(os.path.join(proj.project_dir, name)):
            log("No image file:", name, "skipping ...")
            continue
        if (
            camera.camera_node.getString("make") == "DJI"
            or camera.camera_node.getString("make") == "Hasselblad"
        ):
            # camera is on a gimbal so check if pitch is nearly nadir (-90)
            if pitch_deg > -45:
                log(
                    "gimbal not looking down:",
                    name,
                    "roll:",
                    roll_deg,
                    "pitch:",
                    pitch_deg,
                )
                continue
        elif abs(roll_deg) > max_angle or abs(pitch_deg) > max_angle:
            continue

        base, ext = os.path.splitext(name)
        i1 = image.Image(proj.analysis_dir, base)
        i1.set_aircraft_pose(
            lat_deg, lon_deg, alt_m, yaw_deg, pitch_deg, roll_deg, flight_time
        )
        image_node = images_node.getChild(base, True)
        image_path = os.path.join(meta_dir, base + ".json")
        props_json.save(image_path, image_node)


# for each image, compute the estimated camera pose in NED space from
# the aircraft body pose and the relative camera orientation
def compute_camera_poses(proj):
    """Set camera poses (offset from aircraft pose)."""
    ref_node = getNode("/config/ned_reference", True)
    ref_lat = ref_node.getFloat("lat_deg")
    ref_lon = ref_node.getFloat("lon_deg")
    ref_alt = ref_node.getFloat("alt_m")
    body2cam = camera.get_body2cam()

    for im in proj.image_list:
        ac_pose_node = im.node.getChild("aircraft_pose", True)
        aircraft_lat = ac_pose_node.getFloat("lat_deg")
        aircraft_lon = ac_pose_node.getFloat("lon_deg")
        aircraft_alt = ac_pose_node.getFloat("alt_m")
        ned2body = []
        for i in range(4):
            ned2body.append(ac_pose_node.getFloatEnum("quat", i))

        ned2cam = transformations.quaternion_multiply(ned2body, body2cam)
        (yaw_rad, pitch_rad, roll_rad) = transformations.euler_from_quaternion(
            ned2cam, "rzyx"
        )
        ned = navpy.lla2ned(
            aircraft_lat, aircraft_lon, aircraft_alt, ref_lat, ref_lon, ref_alt
        )
        yd = yaw_rad * r2d
        pd = pitch_rad * r2d
        rd = roll_rad * r2d

        im.set_camera_pose(ned, yd, pd, rd)


# fmt: off
def make_pix4d(
    image_dir,
    force_altitude=None,
    force_heading=None,
    yaw_from_groundtrack=False
):
    """Make a pix4d pose file from project image metadata."""
    if (
        not force_altitude
        and camera.camera_node.getString("make") == "DJI"
        and camera.camera_node.getString("model") in
        ["FC330", "FC6310", "FC6310S"]
    ):
        raise Exception("P4P has unsupported altitude tags")
        # fmt : on

    # load list of images
    files = []
    for file in os.listdir(image_dir):
        if fnmatch.fnmatch(file, "*.jpg") or fnmatch.fnmatch(file, "*.JPG"):
            files.append(file)
    files.sort()

    # save some work if true
    images_have_yaw = False

    images = []
    # read image exif timestamp (and convert to unix seconds)
    for file in files:
        full_name = os.path.join(image_dir, file)
        # print(full_name)
        (
            lon_deg,
            lat_deg,
            alt_m,
            unixtime,
            yaw_deg,
            pitch_deg,
            roll_deg
        ) = exif.get_pose(full_name)

        line = [file, lat_deg, lon_deg]
        if not force_altitude:
            line.append(alt_m)
        else:
            line.append(force_altitude)

        if roll_deg is None:
            line.append(0)  # assume zero roll
        else:
            line.append(roll_deg)
        if pitch_deg is None:
            line.append(0)  # assume zero pitch
        else:
            line.append(pitch_deg)
        if force_heading is not None:
            line.append(force_heading)
        elif yaw_deg is not None:
            images_have_yaw = True
            line.append(yaw_deg)
        else:
            # no yaw info found in metadata
            line.append(0)

        images.append(line)

    if (not force_heading and not images_have_yaw) or yaw_from_groundtrack:
        print("do yaw from ground track")
        # do extra work to estimate yaw heading from gps ground track
        from rcUAS import wgs84

        for i in range(len(images)):
            if i > 0:
                prev = images[i - 1]
            else:
                prev = None
            cur = images[i]
            if i < len(images) - 1:
                next_cam = images[i + 1]
            else:
                next_cam = None

            if prev is not None:
                (prev_hdg, rev_course, prev_dist) = wgs84.geo_inverse(
                    prev[1], prev[2], cur[1], cur[2]
                )
            else:
                prev_hdg = 0.0
                prev_dist = 0.0
            if next_cam is not None:
                (next_hdg, rev_course, next_dist) = wgs84.geo_inverse(
                    cur[1], cur[2], next_cam[1], next_cam[2]
                )
            else:
                next_hdg = 0.0
                next_dist = 0.0

            prev_hdgx = math.cos(prev_hdg * d2r)
            prev_hdgy = math.sin(prev_hdg * d2r)
            next_hdgx = math.cos(next_hdg * d2r)
            next_hdgy = math.sin(next_hdg * d2r)
            avg_hdgx = (prev_hdgx * prev_dist + next_hdgx * next_dist) / (
                prev_dist + next_dist
            )
            avg_hdgy = (prev_hdgy * prev_dist + next_hdgy * next_dist) / (
                prev_dist + next_dist
            )
            avg_hdg = math.atan2(avg_hdgy, avg_hdgx) * r2d
            if avg_hdg < 0:
                avg_hdg += 360.0
            images[i][6] = avg_hdg

    # sanity check
    output_file = os.path.join(image_dir, "pix4d.csv")
    if os.path.exists(output_file):
        log(output_file, "exists, please rename it and rerun this script.")
        quit()
    log("Creating pix4d image pose file:", output_file, "images:", len(files))

    # traverse the image list and create output csv file
    with open(output_file, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "File Name",
                "Lat (decimal degrees)",
                "Lon (decimal degrees)",
                "Alt (meters MSL)",
                "Roll (decimal degrees)",
                "Pitch (decimal degrees)",
                "Yaw (decimal degrees)",
            ],
        )
        writer.writeheader()
        for line in images:
            image = line[0]
            lat_deg = line[1]
            lon_deg = line[2]
            alt_m = line[3]
            roll_deg = line[4]
            pitch_deg = line[5]
            yaw_deg = line[6]
            writer.writerow(
                {
                    "File Name": os.path.basename(image),
                    "Lat (decimal degrees)": "%.10f" % lat_deg,
                    "Lon (decimal degrees)": "%.10f" % lon_deg,
                    "Alt (meters MSL)": "%.2f" % alt_m,
                    "Roll (decimal degrees)": "%.2f" % roll_deg,
                    "Pitch (decimal degrees)": "%.2f" % pitch_deg,
                    "Yaw (decimal degrees)": "%.2f" % yaw_deg,
                }
            )
