import random

from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, LPoint3, NodePath, Texture, TransparencyAttrib
from direct.gui.DirectGui import *

class Annotations():
    def __init__(self, render):
        self.render = render
        random.seed()
        self.icon = loader.loadTexture('explore/marker-icon-2x.png')
        self.view_size = 100
        self.markers = []
        for i in range(20):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            self.markers.append( [x, y] )
        self.nodes = []

    def toggle(self, cam_pos):
        mw = base.mouseWatcherNode
        if not mw.hasMouse():
            return
        props = base.win.getProperties()
        y = props.getYSize()
        pxm = float(y) / self.view_size
        range = 25 / pxm
        mpos = mw.getMouse()
        x = cam_pos[0] + mpos[0] * self.view_size*0.5 * base.getAspectRatio()
        y = cam_pos[1] + mpos[1] * self.view_size*0.5
        # check if we clicked on an existing marker
        exists = False
        for i, m in enumerate(self.markers):
            if abs(x - m[0]) <= range and abs(y - m[1]) <= range:
                exists = True
                del self.markers[i]
                break
        if not exists:
            self.markers.append( [x, y] )
        self.rebuild(self.view_size)
            
    def rebuild(self, view_size):
        self.view_size = view_size
        props = base.win.getProperties()
        y = props.getYSize()
        pxm = float(y) / self.view_size
        hsize = 12 / pxm
        vsize = 40 / pxm
        print(hsize, vsize)
        cm = CardMaker('card')
        cm.setFrame( LPoint3(-hsize, 0,     0 ),
                     LPoint3( hsize, 0,     0 ),
                     LPoint3( hsize, vsize, 0 ),
                     LPoint3(-hsize, vsize, 0 ) )

        for n in self.nodes:
            n.removeNode()

        self.nodes = []
        for m in self.markers:
            node = NodePath(cm.generate())
            node.setTexture(self.icon, 1)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthTest(False)
            node.setDepthWrite(False)
            node.setBin("unsorted", 1)
            node.setPos(m[0], m[1], 0)
            node.reparentTo(self.render)
            self.nodes.append(node)
