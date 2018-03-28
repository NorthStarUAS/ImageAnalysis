#!/usr/bin/python

import os.path

from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
 
class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
 
        # Load the environment model.
        # self.scene1 = self.loader.loadModel("models/environment")
        # self.scene1 = self.loader.loadModel("ex1")
	for n in [ "DSC06368", "DSC06369", "DSC06370", "DSC06371", "DSC06372" ]:
            mn = os.path.join("Textures", n)
            self.model = self.loader.loadModel(mn)
            # Reparent the model to render.
            self.model.reparentTo(self.render)
            # Apply scale and position transforms on the model.
            # self.scene1.setScale(0.25, 0.25, 0.25)
            # self.scene1.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        
    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 12.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20.0 * cos(angleRadians), 600)
        self.camera.setHpr(angleDegrees, -80, 0)
	print angleDegrees
        return Task.cont
 
app = MyApp()
app.run()
