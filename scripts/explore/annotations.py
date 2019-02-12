import csv
import json
import os
import random

import navpy

from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, LPoint3, NodePath, Texture, TransparencyAttrib
from direct.gui.DirectGui import *

class Annotations():
    def __init__(self, render, project_dir, ned_ref):
        self.render = render
        self.project_dir = project_dir
        self.ned_ref = ned_ref
        random.seed()
        self.icon = loader.loadTexture('explore/marker-icon-2x.png')
        self.view_size = 100
        self.markers = []
        # generate some random markers for testing ...
        # for i in range(20):
        #     x = random.uniform(-100, 100)
        #     y = random.uniform(-100, 100)
        #     self.markers.append( [x, y] )
        self.nodes = []
        self.load()

    def load(self):
        file = os.path.join(self.project_dir, 'annotations.json')
        if os.path.exists(file):
            print('Loading saved annotations:', file)
            f = open(file, 'r')
            lla_list = json.load(f)
            f.close()
            for lla in lla_list:
                ned = navpy.lla2ned(lla[0], lla[1], lla[2],
                                    self.ned_ref[0],
                                    self.ned_ref[1],
                                    self.ned_ref[2])
                # print(lla, ned)
                self.markers.append( [ned[1], ned[0]] )
        else:
            print('No annotations file found.')

    def save(self):
        filename = os.path.join(self.project_dir, 'annotations.json')
        print('Saving annotations:', filename)
        lla_list = []
        for m in self.markers:
            lla = navpy.ned2lla( [m[1], m[0], 0.0],
                                 self.ned_ref[0],
                                 self.ned_ref[1],
                                 self.ned_ref[2] )
            lla_list.append(lla)
        f = open(filename, 'w')
        json.dump(lla_list, f, indent=4)
        f.close()

        # write simple csv version
        filename = os.path.join(self.project_dir, 'annotations.csv')
        with open(filename, 'w') as f:
            fieldnames = ['lat_deg', 'lon_deg']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for lla in lla_list:
                writer.writerow({'lat_deg': lla[0], 'lon_deg': lla[1]})

    def toggle(self, cam_pos):
        mw = base.mouseWatcherNode
        if not mw.hasMouse():
            return
        props = base.win.getProperties()
        y = props.getYSize()
        pxm = float(y) / self.view_size
        range = 25 / pxm
        hsize = 12 / pxm
        vsize = 40 / pxm

        mpos = mw.getMouse()
        x = cam_pos[0] + mpos[0] * self.view_size*0.5 * base.getAspectRatio()
        y = cam_pos[1] + mpos[1] * self.view_size*0.5
        # check if we clicked on an existing marker
        exists = False
        for i, m in enumerate(self.markers):
            dx = abs(x - m[0])
            dy = y - m[1]
            if dx <= (hsize*0.5)+1 and y >= m[1]-1 and y <= m[1]+vsize+1:
                exists = True
                del self.markers[i]
                break
        if not exists:
            self.markers.append( [x, y] )
        self.rebuild(self.view_size)
        self.save()
            
    def rebuild(self, view_size):
        self.view_size = view_size
        props = base.win.getProperties()
        y = props.getYSize()
        pxm = float(y) / self.view_size
        hsize = 12 / pxm
        vsize = 40 / pxm
        #print(hsize, vsize)
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
