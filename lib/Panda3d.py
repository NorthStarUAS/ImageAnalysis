# routines to support generating panda3d models

import subprocess
import cv2
import math
import numpy as np
import os

def make_textures(src_dir, project_dir, image_list, resolution=256):
    dst_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(dst_dir):
        print("Notice: creating texture directory =", dst_dir)
        os.makedirs(dst_dir)
    for image in image_list:
        src = os.path.join(src_dir, image.name)
        dst = os.path.join(dst_dir, image.name)
        if not os.path.exists(dst):
            subprocess.run(['convert', '-resize', '%dx%d!' % (resolution, resolution), src, dst])
        
def make_textures_opencv(src_dir, project_dir, image_list, resolution=256):
    dst_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(dst_dir):
        print("Notice: creating texture directory =", dst_dir)
        os.makedirs(dst_dir)
    for image in image_list:
        src = image.image_file
        dst = os.path.join(dst_dir, image.name + '.JPG')
        print(src, '->', dst)
        if not os.path.exists(dst):
            src = cv2.imread(src, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
            height, width = src.shape[:2]
            # downscale image first
            method = cv2.INTER_AREA  # cv2.INTER_AREA
            scale = cv2.resize(src, (0,0),
                               fx=resolution/float(width),
                               fy=resolution/float(height),
                               interpolation=method)
            # convert to hsv color space
            hsv = cv2.cvtColor(scale, cv2.COLOR_BGR2HSV)
            hue,sat,val = cv2.split(hsv)
            # adaptive histogram equalization on 'value' channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            aeq = clahe.apply(val)
            # recombine
            hsv = cv2.merge((hue,sat,aeq))
            # convert back to rgb
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(dst, result)
            print("Texture %dx%d %s" % (resolution, resolution, dst))
            
def generate(proj, group, ref_image=False, src_dir=".", project_dir=".",
             resolution=512 ):
    # make the textures if needed
    make_textures_opencv(src_dir, project_dir, proj.image_list, resolution)
    
    for name in group:
        image = proj.findImageByName(name)
        if len(image.grid_list) == 0:
            continue

        root, ext = os.path.splitext(image.name)
        name = os.path.join( project_dir, "models", root + ".egg" )
        print("EGG file name:", name)

        f = open(name, "w")
        f.write("<CoordinateSystem> { Z-Up }\n\n")
        f.write("<Texture> tex { \"" + image.name + ".JPG\" }\n\n")
        f.write("<VertexPool> surface {\n")

        # this is contructed in a weird way, but we generate the 2d
        # iteration in the same order that the original grid_list was
        # constucted so it works.
        width, height = image.get_size()
        steps = int(math.sqrt(len(image.grid_list))) - 1
        n = 1
        nan_list = []
        for j in range(steps+1):
            for i in range(steps+1):
                v = image.grid_list[n-1]
                if np.isnan(v[0]) or np.isnan(v[1]) or np.isnan(v[2]):
                    v = [0.0, 0.0, 0.0]
                    nan_list.append( (j * (steps+1)) + i + 1 )
                uv = image.distorted_uv[n-1]
                f.write("  <Vertex> %d {\n" % n)
                f.write("    %.2f %.2f %.2f\n" % (v[0], v[1], v[2]))
                f.write("    <UV> { %.5f %.5f }\n" % (uv[0]/float(width), 1.0-uv[1]/float(height)))
                f.write("  }\n")
                n += 1
        f.write("}\n\n")

        f.write("<Group> surface {\n")

        count = 0
        for j in range(steps):
            for i in range(steps):
                c = (j * (steps+1)) + i + 1
                d = ((j+1) * (steps+1)) + i + 1
                if c in nan_list or d in nan_list or (c+1) in nan_list or (d+1) in nan_list:
                    # skip
                    pass
                else:
                    f.write("  <Polygon> {\n")
                    f.write("   <TRef> { tex }\n")
                    f.write("   <Normal> { 0 0 1 }\n")
                    f.write("   <VertexRef> { %d %d %d %d <Ref> { surface } }\n" \
                            % (d, d+1, c+1, c))
                    f.write("  }\n")
                    count += 1

        f.write("}\n")
        f.close()

        if count == 0:
            # uh oh, no polygons fully projected onto the surface for
            # this image.  For now let's warn and delete the model
            print("Warning: no polygons fully on surface, removing:", name)
            os.remove(name)
