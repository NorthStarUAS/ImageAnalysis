#!/usr/bin/python3

import argparse
import fnmatch
import os
import re
from zipfile import ZipFile

from lib import ProjectMgr

parser = argparse.ArgumentParser(description='Compute Delauney triangulation of matches.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--output', required=True, help='specify output /path/file')
args = parser.parse_args()

# sanity check
if not os.path.isdir(args.project):
    print("Cannot find project:", args.project)
    quit()

# load the project
proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

# zip base
base = os.path.basename(args.output)
zip_base, extension = os.path.splitext(base)

# list of files to include in zip archive
file_paths = []
zip_paths = []

def append(full_path):
    file_paths.append(full_path)
    zip_path = re.sub(args.project, '', full_path)
    zip_paths.append(os.path.join(zip_base, zip_path))
    
# project config file
config_json = os.path.join(proj.analysis_dir, 'config.json')
if not os.path.exists(config_json):
    print("Cannot find:", config_json)
    quit()
else:
    append(config_json)

# annotations
annotations_json = os.path.join(proj.analysis_dir, 'annotations.json')
if os.path.exists(annotations_json):
    append(annotations_json)
annotations_csv = os.path.join(proj.analysis_dir, 'annotations.csv')
if os.path.exists(annotations_csv):
    append(annotations_csv)

meta_dir = os.path.join(proj.analysis_dir, 'meta')
if not os.path.isdir(meta_dir):
    print("Cannot find:", meta_dir)
    quit()
else:
    for file in sorted(os.listdir(meta_dir)):
        if fnmatch.fnmatch(file, '*.json'):
            file_path = os.path.join(meta_dir, file)
            append(file_path)

models_dir = os.path.join(proj.analysis_dir, 'models')
if not os.path.isdir(models_dir):
    print("Cannot find:", models_dir)
    quit()
else:
    for file in sorted(os.listdir(models_dir)):
        file_path = os.path.join(models_dir, file)
        append(file_path)

# # create the images directory (of symbolic links to full res images) if needed
# images_dir = os.path.join(args.project, 'images')
# if not os.path.isdir(images_dir):
#     print("Creating project images directory:", images_dir)
#     os.makedirs(images_dir)

# for image in proj.image_list:
#     base_name = os.path.basename(image.image_file)
#     image_file = os.path.join(images_dir, base_name)
#     # print(base_name, image.image_file)
#     if os.path.exists(image_file):
#         if not os.path.islink(image_file):
#             print("Warning:", image_file, "is not a symbolic link")
#     else:
#         print("Linking:", image.image_file, "->", image_file)
#         os.symlink(image.image_file, image_file)

for file in sorted(os.listdir(args.project)):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        file_path = os.path.join(args.project, file)
        append(file_path)

for i in range(len(file_paths)):
    print(file_paths[i], zip_paths[i])

# writing files to a zip file
zipfile = args.output
if extension == '':
    zipfile += '.zip'
print("Writing zip file:", zipfile)

with ZipFile(zipfile, 'w') as zip:
    # writing each file one by one
    for i in range(len(file_paths)):
        print("  adding:", file_paths[i])
        zip.write(file_paths[i], arcname=zip_paths[i])
