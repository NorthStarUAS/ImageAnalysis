import random

from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, LPoint3, NodePath, Texture, TransparencyAttrib
from direct.gui.DirectGui import *

class Annotations():
    def __init__(self, render):
        self.render = render
        random.seed()
        self.icon = loader.loadTexture('explore/marker-icon-2x.png')
        self.markers = []
        for i in range(20):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            self.markers.append( [x, y] )
        self.nodes = []

    def rebuild(self, view_size):
        props = base.win.getProperties()
        y = props.getYSize()
        pxm = float(y) / view_size
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
