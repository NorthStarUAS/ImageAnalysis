# state management module

import os
import time

state_path = None

def init(analysis_path):
    global state_path
    if not os.path.isdir(analysis_path):
        logger.log("state: analysis_path missing:", analysis_path)
    state_path = analysis_path

def check(file):
    full_file = os.path.join(state_path, file)
    if os.path.isfile(full_file):
        return os.path.getmtime(full_file)
    else:
        return 0

def update(file):
    full_file = os.path.join(state_path, file)
    f = open(full_file, "w")
    f.write(str(time.time()) + "\n")
    f.close()

