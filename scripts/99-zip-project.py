#!/usr/bin/python3

import argparse
import fnmatch
import os
from zipfile import ZipFile

from lib import ProjectMgr

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--area', default='area-00', help='sub area directory')
args = parser.parse_args()

# list of files to include in zip archive
file_paths = []

# sanity check
if not os.path.isdir(args.project):
    print("Cannot find project:", args.project)
    quit()

# load the project
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_area_info(args.area)

# project config file
config_json = os.path.join(args.project, 'config.json')
if not os.path.exists(config_json):
    print("Cannot find:", config_json)
    quit()
else:
    file_paths.append(config_json)

# annotations
annotations_json = os.path.join(args.project, 'annotations.json')
if os.path.exists(annotations_json):
    file_paths.append(annotations_json)
annotations_csv = os.path.join(args.project, 'annotations.csv')
if os.path.exists(annotations_csv):
    file_paths.append(annotations_csv)

meta_dir = os.path.join(args.project, 'meta')
if not os.path.isdir(meta_dir):
    print("Cannot find:", meta_dir)
    quit()
else:
    for file in os.listdir(meta_dir):
        if fnmatch.fnmatch(file, '*.json'):
            file_path = os.path.join(meta_dir, file)
            file_paths.append(file_path)

models_dir = os.path.join(args.project, 'models')
if not os.path.isdir(models_dir):
    print("Cannot find:", models_dir)
    quit()
else:
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        file_paths.append(file_path)

# create the images directory (of symbolic links to full res images) if needed
images_dir = os.path.join(args.project, 'images')
if not os.path.isdir(images_dir):
    print("Creating project images directory:", images_dir)
    os.makedirs(images_dir)

for image in proj.image_list:
    base_name = os.path.basename(image.image_file)
    image_file = os.path.join(images_dir, base_name)
    # print(base_name, image.image_file)
    if os.path.exists(image_file):
        if not os.path.islink(image_file):
            print("Warning:", image_file, "is not a symbolic link")
    else:
        print("Linking:", image.image_file, "->", image_file)
        os.symlink(image.image_file, image_file)

for file in os.listdir(images_dir):
    file_path = os.path.join(images_dir, file)
    file_paths.append(file_path)

# writing files to a zip file
zipfile = args.project + '.zip'
print("Writing zip file:", zipfile)
with ZipFile(zipfile, 'w') as zip:
    # writing each file one by one
    for file in file_paths:
        print("  adding:", file)
        zip.write(file)
