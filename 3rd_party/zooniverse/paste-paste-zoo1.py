#!/usr/bin/python3

import argparse
import csv
import cv2
import json
import os

parser = argparse.ArgumentParser(description='Chop up an image for zooniverse.')
parser.add_argument("subjectsets", help="subject set id to local project path lookup")
parser.add_argument('subjects', help='zooniverse subjects CSV file')
parser.add_argument('classifications', help='zooniverse classifications CSV file')
parser.add_argument('--verbose', action='store_true', help='verbose')
args = parser.parse_args()

# given a path and a subject file name, find the original name this
# refers to
def find_image(path, filename):
    base, ext = os.path.splitext(filename)
    if base[-3] != "_":
        print("ERROR, filename doesn't match expected pattern:", filename)
    else:
        root = base[:-3]
        i = int(base[-2])
        j = int(base[-1])
        print(base, root, i, j)
        fulll = os.path.join(path, root + ".jpg")
        fullu = os.path.join(path, root + ".JPG")
        if os.path.isfile(fulll):
            return fulll, i, j
        elif os.path.isfile(fullu):
            return fullu, i, j
        else:
            print("ERROR, cannot determine original file name for:",
                  path, filename)
    return None, -1, -1
    
# build a map of subject_set_id -> local paths
subject_sets = {}
with open(args.subjectsets, 'r') as fsubjset:
    reader = csv.DictReader(fsubjset)
    for row in reader:
        id = int(row["subject_set_id"])
        subject_sets[id] = row["project_path"]
                
# build a map of subject id -> subject details
subjects = {}
with open(args.subjects, 'r') as fsubj:
    reader = csv.DictReader(fsubj)
    for row in reader:
        id = int(row["subject_id"])
        subject_set_id = int(row["subject_set_id"])
        meta = json.loads(row["metadata"])
        if "filename" in meta:
            if subject_set_id in subject_sets:
                #print(id, subject_set_id, meta["filename"])
                subjects[id] = { "subject_set_id": subject_set_id,
                                 "filename": meta["filename"] }
            else:
                if args.verbose:
                    print("unknown subject set id:", subject_set_id, "ignoring classsification")

by_image = {}
# traverse the classifications and do the stuff
with open(args.classifications, 'r') as fclass:
    reader = csv.DictReader(fclass)
    for row in reader:
        #print(row["classification_id"])
        #print(row["user_name"])
        #print(row["user_id"])
        #print(row["user_ip"])
        #print(row["workflow_id"])
        #print(row["workflow_name"])
        #print(row["workflow_version"])
        #print(row["created_at"])
        #print(row["gold_standard"])
        #print(row["expert"])
        #print(row["metadata"])
        #print(row["annotations"])
        #print(row["subject_data"])
        #print(row["subject_ids"])
        subject_id = int(row["subject_ids"])
        if not subject_id in subjects:
            continue
        subject_set_id = int(subjects[subject_id]["subject_set_id"])
        filename = subjects[subject_id]["filename"]
        if not subject_set_id in subject_sets:
            continue
        project_path = subject_sets[subject_set_id]
        print(subject_id, subject_set_id, project_path, filename)
        fullpath, i, j = find_image(project_path, filename)
        if not fullpath in by_image:
            by_image[fullpath] = []
        meta = json.loads(row["metadata"])
        #print(meta)
        subject_dim = meta["subject_dimensions"][0]
        if subject_dim is None:
            continue
        print(subject_dim["naturalWidth"], subject_dim["naturalHeight"])
        subj_w = subject_dim["naturalWidth"]
        subj_h = subject_dim["naturalHeight"]
        base_w = subj_w * i
        base_h = subj_h * j
        tasks = json.loads(row["annotations"])
        task = tasks[0]
        for i, val in enumerate(task["value"]):
            print(i, val)
            x = round(val["x"])
            y = round(val["y"])
            if "r" in val:
                r = round(val["r"])
            else:
                r = 1
            print(x, y, r)
            deets = val["details"]
            density = deets[0]["value"]
            if len(deets) >= 2:
                confidence = deets[1]["value"]
            if len(deets) >= 3:
                comment = deets[2]["value"]
                if len(comment):
                    print("comment:", comment)
            u = base_w + x
            v = base_h + y
            by_image[fullpath].append( [u, v] )

for fullpath in sorted(by_image.keys()):
    green = (0, 255, 0)
    scale = 0.4
    print(fullpath)
    pt_list = by_image[fullpath]
    rgb = cv2.imread(fullpath, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
    for pt in pt_list:
        cv2.circle(rgb, (pt[0], pt[1]), 20, green, -1)
    preview = cv2.resize(rgb, None, fx=scale, fy=scale)
    h, w = preview.shape[:2]
    print(w, h)
    for i in range(int(w/4), w, int(w/4)):
        cv2.line(preview, (i, 0), (i, h-1), (0, 0, 0), 2)
    for i in range(int(h/4), h, int(h/4)):
        cv2.line(preview, (0, i), (w-1, i), (0, 0, 0), 2)
    cv2.imshow("debug", preview)
    cv2.waitKey()
    
