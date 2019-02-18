#!/usr/bin/python3

from __future__ import print_function
import argparse
import cv2
import fnmatch
import os.path
from progress.bar import Bar
import sys
import time

from props import getNode

import math
import numpy as np

# pip3 install --pre --extra-index-url https://archive.panda3d.org/ panda3d
# pip3 install panda3d

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import LineSegs, NodePath, OrthographicLens, PNMImage, Texture, Filename
from direct.gui.DirectGui import YesNoDialog

from lib import ProjectMgr
from explore import annotations
from explore import reticle

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', help='project directory')
args = parser.parse_args()

if False:
    import wx
    def get_path(wildcard):
        app = wx.App(None)
        style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
        else:
            path = None
        dialog.Destroy()
        return path
    get_path("*")
    print(get_path('*.txt'))
    quit()

if not args.project:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    # file_path = filedialog.askopenfilename()
    file_path = filedialog.askdirectory(title="Please open the project directory", mustexist=True)
    # print('selected:', type(file_path), len(file_path), file_path)
    if file_path:
        args.project = file_path
    else:
        print("no project selected, exiting.")
        quit()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ned_ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]

tcache = {}

# adaptive equalizer
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
 
        # Load the environment model.
        # self.scene1 = self.loader.loadModel("models/environment")
        self.models = []
        self.base_textures = []

        # we would like an orthographic lens
        self.lens = OrthographicLens()
        self.lens.setFilmSize(20, 15)
        base.camNode.setLens(self.lens)

        self.cam_pos = [ 0.0, 0.0, 1000.0 ]
        self.camera.setPos(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        self.camera.setHpr(0, -90.0, 0)
        self.view_size = 100.0
        self.last_ysize = 0
        
        self.top_image = 0

        # modules
        self.annotations = annotations.Annotations(self.render, args.project,
                                                   ned_ref)
        self.reticle = reticle.Reticle(self.render)

        #self.messenger.toggleVerbose()

        # event handlers
        self.accept('arrow_left', self.cam_move, [-0.1, 0, 0])
        self.accept('arrow_right', self.cam_move, [0.1, 0, 0])
        self.accept('arrow_down', self.cam_move, [0, -0.1, 0])
        self.accept('arrow_up', self.cam_move, [0, 0.1, 0])
        self.accept('=', self.cam_zoom, [1.1])
        self.accept('shift-=', self.cam_zoom, [1.1])
        self.accept('-', self.cam_zoom, [1.0/1.1])
        self.accept('wheel_up', self.cam_zoom, [1.1])
        self.accept('wheel_down', self.cam_zoom, [1.0/1.1])
        self.accept('mouse1', self.mouse_state, [0, 1])
        self.accept('mouse1-up', self.mouse_state, [0, 0])
        self.accept('0', self.image_select, [0])
        self.accept('1', self.image_select, [1])
        self.accept('2', self.image_select, [2])
        self.accept('3', self.image_select, [3])
        self.accept('4', self.image_select, [4])
        self.accept('5', self.image_select, [5])
        self.accept('6', self.image_select, [6])
        self.accept('7', self.image_select, [7])
        self.accept('8', self.image_select, [8])
        self.accept('9', self.image_select, [9])
        self.accept('escape', self.quit)
        self.accept('mouse3', self.annotations.toggle, [self.cam_pos])

        # mouse state
        self.last_mpos = [0, 0]
        self.mouse = [0, 0, 0]
        self.last_mouse = [0, 0, 0] 
       
        # Add the tasks to the task manager.
        self.taskMgr.add(self.updateCameraTask, "updateCameraTask")

        self.query_capabilities(display=True)
        
    def query_capabilities(self, display=True):
        gsg=base.win.getGsg()
        print("driver vendor", gsg.getDriverVendor())
        self.driver_vendor = gsg.getDriverVendor()
        print("alpha_scale_via_texture", bool(gsg.getAlphaScaleViaTexture()))
        print("color_scale_via_lighting", bool(gsg.getColorScaleViaLighting()))
        print("copy_texture_inverted", bool(gsg.getCopyTextureInverted()))
        print("max_3d_texture_dimension", gsg.getMax3dTextureDimension())
        print("max_clip_planes", gsg.getMaxClipPlanes())
        print("max_cube_map_dimension", gsg.getMaxCubeMapDimension())
        print("max_lights", gsg.getMaxLights())
        print("max_texture_dimension", gsg.getMaxTextureDimension())
        self.max_texture_dimension = gsg.getMaxTextureDimension()
        print("max_texture_stages", gsg.getMaxTextureStages())
        print("max_vertex_transform_indices",  gsg.getMaxVertexTransformIndices())
        print("max_vertex_transforms", gsg.getMaxVertexTransforms())
        print("shader_model", gsg.getShaderModel())
        print("supports_3d_texture", bool(gsg.getSupports3dTexture()))
        print("supports_basic_shaders", bool(gsg.getSupportsBasicShaders()))
        print("supports_compressed_texture",  bool(gsg.getSupportsCompressedTexture()))
        print("supports_cube_map", bool(gsg.getSupportsCubeMap()))
        print("supports_depth_stencil", bool(gsg.getSupportsDepthStencil()))
        print("supports_depth_texture",  bool(gsg.getSupportsDepthTexture()))
        print("supports_generate_mipmap",  bool(gsg.getSupportsGenerateMipmap()))
        #print("supports_render_texture", bool(gsg.getSupportsRenderTexture()))
        print("supports_shadow_filter", bool(gsg.getSupportsShadowFilter()))
        print("supports_tex_non_pow2", bool(gsg.getSupportsTexNonPow2()))
        self.needs_pow2 = not bool(gsg.getSupportsTexNonPow2())
        if self.driver_vendor == 'Intel' and os.name == 'nt':
            # windows driver lies!
            self.needs_pow2 = True
        print("supports_texture_combine", bool(gsg.getSupportsTextureCombine()))
        print("supports_texture_dot3", bool(gsg.getSupportsTextureDot3()))
        print("supports_texture_saved_result",  bool(gsg.getSupportsTextureSavedResult()))
        print("supports_two_sided_stencil",  bool(gsg.getSupportsTwoSidedStencil()))
        print("max_vertices_per_array", gsg.getMaxVerticesPerArray())
        print("max_vertices_per_primitive", gsg.getMaxVerticesPerPrimitive())
        print("supported_geom_rendering", gsg.getSupportedGeomRendering())
        print("supports_multisample", bool(gsg.getSupportsMultisample()))
        print("supports_occlusion_query", bool(gsg.getSupportsOcclusionQuery()))
        print("prefers_triangle_strips", bool(gsg.prefersTriangleStrips()))

    def tmpItemSel(self, arg):
        self.dialog.cleanup()
        print('result:', arg)
        
    def dialog_test(self, tmp):
        self.dialog = YesNoDialog(dialogName="YesNoCancelDialog", text="Please choose:", command=self.tmpItemSel)
        
    def mouse_state(self, index, state):
        self.mouse[index] = state
    
    def pretty_print(self, node, indent=''):
        for child in node.getChildren():
            print(indent, child)
                
    def load(self, path):
        files = []
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*.egg'):
                # print('load:', file)
                files.append(file)
        bar = Bar('Loading models:', max=len(files))
        for file in files:
            # load and reparent each egg file
            pandafile = Filename.fromOsSpecific(os.path.join(path, file))
            model = self.loader.loadModel(pandafile)

            # print(file)
            # self.pretty_print(model, '  ')
            
            model.reparentTo(self.render)
            self.models.append(model)
            tex = model.findTexture('*')
            if tex != None:
                tex.setWrapU(Texture.WM_clamp)
                tex.setWrapV(Texture.WM_clamp)
            else:
                print('Oops, no texture found for:', file)
            self.base_textures.append(tex)
            bar.next()
        bar.finish()
        self.sortImages()
        self.annotations.rebuild(self.view_size)

    def cam_move(self, x, y, z, sort=True):
        #print('move:', x, y)
        self.cam_pos[0] += x * self.view_size * base.getAspectRatio()
        self.cam_pos[1] += y * self.view_size
        if sort:
            self.image_select(0)
            self.sortImages()
        
    def cam_zoom(self, f):
        self.view_size /= f
        self.annotations.rebuild(self.view_size)

    def quit(self):
        quit()

    def image_select(self, level):
        self.top_image = level
        self.sortImages()
        
    # Define a procedure to move the camera.
    def updateCameraTask(self, task):
        self.camera.setPos(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        self.camera.setHpr(0, -90, 0)
        self.lens.setFilmSize(self.view_size*base.getAspectRatio(),
                              self.view_size)
        # reticle
        self.reticle.update(self.cam_pos, self.view_size)
        
        # annotations
        props = base.win.getProperties()
        y = props.getYSize()
        if y != self.last_ysize:
            self.annotations.rebuild(self.view_size)
            self.last_ysize = y
        mw = base.mouseWatcherNode
        if mw.hasMouse():
            mpos = mw.getMouse()
            if self.mouse[0]:
                dx = self.last_mpos[0] - mpos[0]
                dy = self.last_mpos[1] - mpos[1]
                self.cam_move( dx * 0.5, dy * 0.5, 0, sort=False)
            elif not self.mouse[0] and self.last_mouse[0]:
                # button up
                self.cam_move( 0, 0, 0, sort=True)
            self.last_mpos = list(mpos)
            self.last_mouse[0] = self.mouse[0]
        return Task.cont

    # return true if cam_pos inside bounding corners
    def inbounds(self, b):
        if self.cam_pos[0] < b[0][0] or self.cam_pos[0] > b[1][0]:
            return False
        elif self.cam_pos[1] < b[0][1] or self.cam_pos[1] > b[1][1]:
            return False
        else:
            return True
        
    def sortImages(self):
        # sort images by (hopefully) best covering view center
        result_list = []
        for m in self.models:
            b = m.getTightBounds()
            #print('tight', b)
            if b:
                center = [ (b[0][0] + b[1][0]) * 0.5,
                           (b[0][1] + b[1][1]) * 0.5,
                           (b[0][2] + b[1][2]) * 0.5 ]
                vol = [ b[0][0] - b[1][0],
                        b[0][1] - b[1][1],
                        b[0][2] - b[1][2] ]
                span = math.sqrt(vol[0]*vol[0] + vol[1]*vol[1] + vol[2]*vol[2])
                dx = center[0] - self.cam_pos[0]
                dy = center[1] - self.cam_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                #print('center:', center, 'span:', span, 'dist:', dist)
                metric = dist + (span * 0.1)
                if not self.inbounds(b):
                    metric += 1000
                result_list.append( [metric, m] )
        result_list = sorted(result_list, key=lambda fields: fields[0],
                             reverse=True)
        top = result_list[-1-self.top_image][1]
        top.setColor(1.0, 1.0, 1.0, 1.0)
        self.updateTexture(top)

        for i, line in enumerate(result_list):
            m = line[1]
            if m == top:
                m.setBin("fixed", 2*len(self.models))
            elif m.getName() in tcache:
                # reward draw order for models with high res texture loaded
                m.setBin("fixed", i + len(self.models))
            else:
                m.setBin("fixed", i)
            m.setDepthTest(False)
            m.setDepthWrite(False)
            if m != top:
                m.setColor(0.8, 0.8, 0.8, 1.0)

    def updateTexture(self, main):
        dir_node = getNode('/config/directories', True)
        
        # reset base textures
        for i, m in enumerate(self.models):
            if m != main:
                if m.getName() in tcache:
                    fulltex = tcache[m.getName()][1]
                    self.models[i].setTexture(fulltex, 1)
                else:
                    if self.base_textures[i] != None:
                        self.models[i].setTexture(self.base_textures[i], 1)
            else:
                print(m.getName())
                if m.getName() in tcache:
                    fulltex = tcache[m.getName()][1]
                    self.models[i].setTexture(fulltex, 1)
                    continue
                base, ext = os.path.splitext(m.getName())
                image_file = None
                search = [ os.path.join(args.project, 'images') ]
                for i in range( dir_node.getLen('image_sources') ):
                    dir = dir_node.getStringEnum('image_sources', i)
                    search.append(dir)
                for dir in search:
                    tmp1 = os.path.join(dir, base + '.JPG')
                    tmp2 = os.path.join(dir, base + '.jpg')
                    if os.path.isfile(tmp1):
                        image_file = tmp1
                    elif os.path.isfile(tmp2):
                        image_file = tmp2
                if not image_file:
                    print('Warning: no full resolution image source file found:', base)
                else:
                    if True:
                        # example of passing an opencv image as a
                        # panda texture, except currently only works
                        # for gray scale (need to find the proper
                        # constant for rgb in setup2dTexture()
                        print(base, image_file)
                        #image = proj.findImageByName(base)
                        #print(image)
                        rgb = cv2.imread(image_file, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
                        rgb = np.flipud(rgb)
                        h, w = rgb.shape[:2]
                        print('shape: (%d,%d)' % (w, h))
                        rescale = False
                        if h > self.max_texture_dimension:
                            h = self.max_texture_dimension
                            rescale = True
                        if w > self.max_texture_dimension:
                            w = self.max_texture_dimension
                            rescale = True
                        if self.needs_pow2:
                            h2 = 2**math.floor(math.log(h,2))
                            w2 = 2**math.floor(math.log(w,2))
                            if h2 != h:
                                h = h2
                                rescale = True
                            if w2 != w:
                                w = w2
                                rescale = True
                        if rescale:
                            print("Notice: rescaling texture to (%d,%d) to honor video card capability." % (w, h))
                            rgb = cv2.resize(rgb, (w,h))
                            
                        # equalize
                        if False:
                            # equalize val (essentially gray scale level)
                            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                            hue, sat, val = cv2.split(hsv)
                            aeq = clahe.apply(val)
                            # recombine
                            hsv = cv2.merge((hue,sat,aeq))
                            # convert back to rgb
                            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        else:
                            # equalize individual b, g, r channels
                            b, g, r = cv2.split(rgb)
                            b = clahe.apply(b)
                            g = clahe.apply(g)
                            r = clahe.apply(r)
                            result = cv2.merge((b,g,r))
                        fulltex = Texture(base)
                        fulltex.setCompression(Texture.CMOff)
                        fulltex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgb)
                        fulltex.setRamImage(result)
                        # fulltex.load(rgb) # for loading a pnm image
                        fulltex.setWrapU(Texture.WM_clamp)
                        fulltex.setWrapV(Texture.WM_clamp)
                        m.setTexture(fulltex, 1)
                        tcache[m.getName()] = [m, fulltex, time.time()]
                    else:
                        print(image_file)
                        fulltex = self.loader.loadTexture(image_file)
                        fulltex.setWrapU(Texture.WM_clamp)
                        fulltex.setWrapV(Texture.WM_clamp)
                        #print('fulltex:', fulltex)
                        m.setTexture(fulltex, 1)
                        tcache[m.getName()] = [m, fulltex, time.time()]
        cachesize = 10
        while len(tcache) > cachesize:
            oldest_time = time.time()
            oldest_name = ""
            for name in tcache:
                if tcache[name][2] < oldest_time:
                    oldest_time = tcache[name][2]
                    oldest_name = name
            del tcache[oldest_name]
    
app = MyApp()
app.load( os.path.join(args.project, "models") )
app.run()
