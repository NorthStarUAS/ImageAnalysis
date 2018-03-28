#!/usr/bin/python

import argparse
import fnmatch
import os.path
import sys

from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

ref = proj.ned_reference_lla

class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
 
        # Load the environment model.
        # self.scene1 = self.loader.loadModel("models/environment")

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

    def load(self, path):
        files = []
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*.egg'):
                files.append(file)
	for file in files:
            # load and reparent each egg file
            self.model = self.loader.loadModel(os.path.join(path, file))
            self.model.reparentTo(self.render)
        
    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 12.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20.0 * cos(angleRadians), 600)
        self.camera.setHpr(angleDegrees, -80, 0)
	print angleDegrees
        return Task.cont
 
app = MyApp()
app.load( os.path.join(args.project, "Textures") )
app.run()
