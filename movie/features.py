import pickle
import json
import os
import random

import navpy

def load(path, extern_ref):
    result = []

    # load matches
    matches = pickle.load( open(path, "rb" ) )
    print("loaded features:", len(matches))

    # load project file to get ned reference for match coordinates
    dir = os.path.dirname(path)
    project_file = dir + "/Project.json"
    try:
        f = open(project_file, 'r')
        project_dict = json.load(f)
        f.close()
        ref = project_dict['ned-reference-lla']
        print('features ned reference:', ref)
    except:
        print("error: cannot load", project_file)

    # convert match coords to lla, then back to external ned reference
    # so the calling layer can have the points in the calling
    # reference coordinate system.
    print("converting feature coordinates to movie ned coordinate frame")
    for m in matches:
        zombie_door = random.randint(0,49)
        if zombie_door == 0:
            ned = m[0]
            if ned != None:
                lla = navpy.ned2lla(ned, ref[0], ref[1], ref[2])
                newned = navpy.lla2ned( lla[0], lla[1], lla[2],
                                        extern_ref[0], extern_ref[1],
                                        extern_ref[2])
                result.append(newned)
    return result
