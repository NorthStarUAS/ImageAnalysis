#!/usr/bin/python

import argparse
import fnmatch
import os.path
from progress.bar import Bar
import sys

from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import OrthographicLens

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
        self.models = []

        # we would like an orthographic lens
        self.lens = OrthographicLens()
        self.lens.setFilmSize(20, 15)
        base.camNode.setLens(self.lens)

        self.cam_pos = [ ref[1], ref[0], -ref[2] + 1000 ]
        self.camera.setPos(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        self.camera.setHpr(0, -89.9, 0)
        self.view_size = 100.0

        # setup keyboard handlers
        #self.messenger.toggleVerbose()
        self.accept('arrow_left', self.move_left)
        self.accept('arrow_right', self.move_right)
        self.accept('arrow_down', self.move_down)
        self.accept('arrow_up', self.move_up)
        self.accept('=', self.move_closer)
        self.accept('shift-=', self.move_closer)
        self.accept('-', self.move_further)
        
        # Add the updateCameraTask procedure to the task manager.
        self.taskMgr.add(self.updateCameraTask, "updateCameraTask")

    def load(self, path):
        files = []
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*.egg'):
                files.append(file)
        bar = Bar('Loading textures:', max=len(files))
	for file in files:
            # load and reparent each egg file
            model = self.loader.loadModel(os.path.join(path, file))
            model.reparentTo(self.render)
            self.models.append(model)
            bar.next()
        bar.finish()

    def move_left(self):
        self.cam_pos[0] -= self.view_size / 10.0
    def move_right(self):
        self.cam_pos[0] += self.view_size / 10.0
    def move_down(self):
        self.cam_pos[1] -= self.view_size / 10.0
    def move_up(self):
        self.cam_pos[1] += self.view_size / 10.0
    def move_closer(self):
        self.cam_pos[2] -= 10
        self.view_size /= 1.1
    def move_further(self):
        self.cam_pos[2] += 10
        self.view_size *= 1.1
        
    # Define a procedure to move the camera.
    def updateCameraTask(self, task):
        print(base.getAspectRatio())
        self.camera.setPos(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        self.camera.setHpr(0, -90, 0)
        self.lens.setFilmSize(self.view_size*base.getAspectRatio(),
                              self.view_size)
        return Task.cont
    
app = MyApp()
app.load( os.path.join(args.project, "Textures") )
app.run()
