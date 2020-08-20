#!/usr/bin/env python3

import argparse
import fnmatch
import os
import re
from zipfile import ZipFile

from lib import project

parser = argparse.ArgumentParser(description='Make a redistributable zip archive of a project.')
parser.add_argument('project', help='project directory')
parser.add_argument('--output', required=True, help='specify output /path/file')
args = parser.parse_args()

# sanity check
if not os.path.isdir(args.project):
    print("Cannot find project:", args.project)
    quit()

# load the project
proj = project.ProjectMgr(args.project)
proj.load_images_info()

# zip base
base = os.path.basename(args.output)
zip_base, extension = os.path.splitext(base)
print("zip_base:", zip_base)

# list of files to include in zip archive
file_paths = []
zip_paths = []

def append(full_path):
    file_paths.append(full_path)
    zip_path = re.sub(args.project, '', full_path)
    zip_path = zip_path.lstrip('/')
    #print("zip:", zip_base, zip_path, 'full:', os.path.join(zip_base, zip_path))
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
annotations_kml = os.path.join(proj.analysis_dir, 'annotations.kml')
if os.path.exists(annotations_kml):
    append(annotations_kml)

# histogram
hist_file = os.path.join(proj.analysis_dir, 'histogram')
if os.path.exists(hist_file):
    append(hist_file)

# smart
smart_json = os.path.join(proj.analysis_dir, 'smart.json')
if os.path.exists(smart_json):
    append(smart_json)
    
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

for file in sorted(os.listdir(args.project)):
    if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
        file_path = os.path.join(args.project, file)
        append(file_path)
    if file == 'pix4d.csv' or file == 'image-metadata.txt':
        file_path = os.path.join(args.project, file)
        append(file_path)

# for i in range(len(file_paths)):
#     print(file_paths[i], zip_paths[i])

# writing files to a zip file
zipfile = args.output
if extension == '':
    zipfile += '.zip'
print("Writing zip file:", zipfile)

with ZipFile(zipfile, 'w') as zip:
    # writing each file one by one
    for i in range(len(file_paths)):
        print("  adding:", file_paths[i], "(zip)", zip_paths[i])
        zip.write(file_paths[i], arcname=zip_paths[i])
