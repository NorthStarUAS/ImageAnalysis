#!/usr/bin/env python3

import argparse
import csv
import fnmatch
import json
import os
import shutil

from props import getNode
import props_json

parser = argparse.ArgumentParser(description='Create an group project.')
parser.add_argument('project', help='Directory with a set of aerial images.')
parser.add_argument('source', metavar='src-project', nargs='+',
                    help='input projects')
args = parser.parse_args()

project_dir = args.project
analysis_dir = os.path.join(project_dir, "ImageAnalysis")
meta_dir = os.path.join(analysis_dir, "meta")
models_dir = os.path.join(analysis_dir, "models")
state_dir = os.path.join(analysis_dir, "state")
cache_dir = os.path.join(analysis_dir, "cache")

if not os.path.isdir(project_dir):
    os.makedirs(project_dir)
else:
    print("Project directory already exists:", project_dir)
    print("aborting ...")
    quit()

if not os.path.isdir(analysis_dir):
    os.makedirs(analysis_dir)

if not os.path.isdir(meta_dir):
    os.makedirs(meta_dir)

if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

if not os.path.isdir(state_dir):
    os.makedirs(state_dir)

if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

# quick input sanity check
for p in args.source:
    if not os.path.isdir(p):
        print("Cannot find source project:", p)
        print("Aborting.")
        quit()

# add symbolic links to source images from each project
print("Creating symbolic links to source image files...")
for p in args.source:
    for image in sorted(os.listdir(p)):
        if fnmatch.fnmatch(image, '*.jpg') or fnmatch.fnmatch(image, '*.JPG'):
            src = os.path.join(p, image)
            dest = os.path.join(project_dir, image)
            if os.path.exists(dest):
                print("Warning, dest already exists:", dest)
            else:
                os.symlink(src, dest)

# create a combo pix4d.csv file
print("Assembling combination pix4d.csv file")
full_list = []
fields = None
for p in args.source:
    csv_path = os.path.join(p, "pix4d.csv")
    if not os.path.exists(csv_path):
        break
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if fields is None:
                fields = row.keys()
            full_list.append(row)
if len(full_list):
    csv_path = os.path.join(project_dir, "pix4d.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in full_list:
            writer.writerow(row)

print("Assembling combination image-metadata.txt file")
full_list = []
fields = None
for p in args.source:
    csv_path = os.path.join(p, "image-metadata.txt")
    if not os.path.exists(csv_path):
        break
    f = open(csv_path, "r")
    for row in f:
        full_list.append(row)
if len(full_list):
    csv_path = os.path.join(project_dir, "image-metadata.txt")
    f = open(csv_path, "w")
    for row in full_list:
        f.write(row)

# copy config.json from first listed source project
print("Copying config.json from source project")
config_src = os.path.join(args.source[0], "ImageAnalysis", "config.json")
config_dest = os.path.join(analysis_dir, "config.json")
if os.path.exists(config_src):
    shutil.copyfile(config_src, config_dest)

# assemble the collective smart.json file
print("Combining all the smart.json files")
smart_node = getNode("/smart", True)
for p in args.source:
    smart_src = os.path.join(p, "ImageAnalysis", "smart.json")
    props_json.load(smart_src, smart_node)
smart_dst = os.path.join(project_dir, "ImageAnalysis", "smart.json")
props_json.save(smart_dst, smart_node)

# assemble the collective annotations.json file
print("Combining any annotations.json files")
ann_node = getNode("/annotations", True)
ann_list = []
for p in args.source:
    ann_src = os.path.join(p, "ImageAnalysis", "annotations.json")
    if os.path.exists(ann_src):
        f = open(ann_src, 'r')
        ann_list += json.load(f)
ann_dst = os.path.join(project_dir, "ImageAnalysis", "annotations.json")
f = open(ann_dst, 'w')
json.dump(ann_list, f, indent=4)

# populate the meta directory
print("Populating the meta directory with symbolic links.")
for p in args.source:
    meta_src = os.path.join(p, "ImageAnalysis", "meta")
    for file in sorted(os.listdir(meta_src)):
        if fnmatch.fnmatch(file, '*.json') or fnmatch.fnmatch(file, '*.match'):
            src = os.path.join(meta_src, file)
            dest = os.path.join(meta_dir, file)
            shutil.copyfile(src, dest)

# populate the cache directory
print("Populating the cache directory with symbolic links.")
for p in args.source:
    cache_src = os.path.join(p, "ImageAnalysis", "cache")
    for file in sorted(os.listdir(cache_src)):
        if fnmatch.fnmatch(file, '*.feat') or fnmatch.fnmatch(file, '*.desc'):
            src = os.path.join(cache_src, file)
            dest = os.path.join(cache_dir, file)
            if os.path.exists(dest):
                print("Warning, dest already exists:", dest)
            else:
                os.symlink(src, dest)

# populate the models directory
print("Populating the models directory with symbolic links.")
for p in args.source:
    models_src = os.path.join(p, "ImageAnalysis", "models")
    for file in sorted(os.listdir(models_src)):
        if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
            src = os.path.join(models_src, file)
            dest = os.path.join(models_dir, file)
            if os.path.exists(dest):
                print("Warning, dest already exists:", dest)
            else:
                os.symlink(src, dest)

print()
print("Now you can run the process.py script on the new project!")
