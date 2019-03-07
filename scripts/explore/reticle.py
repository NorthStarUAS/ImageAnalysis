# draw a reticle on screen as a 3d object that is kept in line with
# camera center

import math

from panda3d.core import LineSegs, NodePath, TextNode
from direct.gui.OnscreenText import OnscreenText

import navpy

class Reticle():
    def __init__(self, render, surface, ned_ref):
        self.render = render
        self.surface = surface
        self.ned_ref = ned_ref
        self.last_cam_pos = [0, 0, 0]

    def build(self, cam_pos, view_size):
        alpha = 0.6
        depth = 0
        a1 = view_size / 20
        a2 = view_size / 5
        props = base.win.getProperties()
        y = props.getYSize()
        pxm = float(y) / view_size
        text_scale = 42 / y
        # print("y:", y, "view_size:", view_size, "pxm:", pxm)

        # center reticle
        ls = LineSegs()
        ls.setThickness(1)
        ls.setColor(0.0, 1.0, 0.0, alpha)
        ls.moveTo(cam_pos[0] + a1, cam_pos[1], 0)
        ls.drawTo(cam_pos[0] + a2, cam_pos[1], 0)
        ls.moveTo(cam_pos[0] - a1, cam_pos[1], 0)
        ls.drawTo(cam_pos[0] - a2, cam_pos[1], 0)
        ls.moveTo(cam_pos[0], cam_pos[1] + a1, 0)
        ls.drawTo(cam_pos[0], cam_pos[1] + a2, 0)
        ls.moveTo(cam_pos[0], cam_pos[1] - a1, 0)
        ls.drawTo(cam_pos[0], cam_pos[1] - a2, 0)
        self.node = NodePath(ls.create())
        self.node.setDepthTest(False)
        self.node.setDepthWrite(False)
        self.node.setBin("unsorted", depth)
        self.node.reparentTo(self.render)

        # measurement marker
        h_size = view_size * base.getAspectRatio()
        h = math.pow(2, int(round(math.log2(h_size/10.0))))
        # print("h_size:", h_size, h)
        ls = LineSegs()
        ls.setThickness(2)
        ls.setColor(0.0, 1.0, 0.0, alpha)
        ls.moveTo(cam_pos[0]-0.48*h_size, cam_pos[1]-0.48*view_size, 0)
        ls.drawTo(cam_pos[0]-0.48*h_size + h, cam_pos[1]-0.48*view_size, 0)
        ls.moveTo(cam_pos[0]-0.48*h_size, cam_pos[1]-0.48*view_size, 0)
        ls.drawTo(cam_pos[0]-0.48*h_size, cam_pos[1]-0.46*view_size, 0)
        ls.moveTo(cam_pos[0]-0.48*h_size + h, cam_pos[1]-0.48*view_size, 0)
        ls.drawTo(cam_pos[0]-0.48*h_size + h, cam_pos[1]-0.46*view_size, 0)
        self.node1 = NodePath(ls.create())
        self.node1.setDepthTest(False)
        self.node1.setDepthWrite(False)
        self.node1.setBin("unsorted", depth)
        self.node1.reparentTo(self.render)
        if h >= 1.0:
            dist_text = "%.0f m" % (h)
        elif h >= 0.1:
            dist_text = "%.1f cm" % (h * 100)
        else:
            dist_text = "%.1f mm" % (h * 1000)
        self.text2 = OnscreenText(text=dist_text,
                                  pos=(-0.95*base.getAspectRatio(), -0.94),
                                  scale=text_scale,
                                  fg=(0.0, 1.0, 0.0, 1.0),
                                  shadow=(0.1, 0.1, 0.1, 0.8),
                                  align=TextNode.ALeft)

        # position display
        z = self.surface.get_elevation(cam_pos[1], cam_pos[0])
        lla = navpy.ned2lla( [cam_pos[1], cam_pos[0], z],
                             self.ned_ref[0], self.ned_ref[1], self.ned_ref[2] )
        pos_str = "Lat: %.7f  Lon: %.7f  Alt(m): %.1f" % (lla[0], lla[1], lla[2])
        self.text1 = OnscreenText(text=pos_str,
                                  pos=(0.95*base.getAspectRatio(), -0.95),
                                  scale=text_scale,
                                  fg=(0.0, 1.0, 0.0, 1.0),
                                  shadow=(0.1, 0.1, 0.1, 0.8),
                                  align=TextNode.ARight)
        
    def delete(self):
        if hasattr(self, 'node'):
            self.node.removeNode()
        if hasattr(self, 'node1'):
            self.node1.removeNode()
        if hasattr(self, 'text1'):
            self.text1.destroy()
        if hasattr(self, 'text2'):
            self.text2.destroy()
     
    def update(self, cam_pos, view_size):
        self.delete()
        self.build(cam_pos, view_size)
