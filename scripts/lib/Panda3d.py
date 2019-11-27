# routines to support generating panda3d models

import subprocess
import cv2
import math
import numpy as np
import os
import scipy.spatial

def make_textures(src_dir, analysis_dir, image_list, resolution=512):
    dst_dir = os.path.join(analysis_dir, 'models')
    if not os.path.exists(dst_dir):
        print("Notice: creating models directory =", dst_dir)
        os.makedirs(dst_dir)
    for image in image_list:
        src = os.path.join(src_dir, image.name)
        dst = os.path.join(dst_dir, image.name)
        if not os.path.exists(dst):
            subprocess.run(['convert', '-resize', '%dx%d!' % (resolution, resolution), src, dst])
        
def make_textures_opencv(src_dir, analysis_dir, image_list, resolution=512):
    dst_dir = os.path.join(analysis_dir, 'models')
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
            do_equalize = False
            if do_equalize:
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
            else:
                result = scale
            cv2.imwrite(dst, result)
            print("Texture %dx%d %s" % (resolution, resolution, dst))
    # make the dummy.jpg image from the first texture
    #src = os.path.join(dst_dir, image_list[0].image_file)
    src = image_list[0].image_file
    dst = os.path.join(dst_dir, "dummy.jpg")
    print("Dummy:", src, dst)
    if not os.path.exists(dst):
        src = cv2.imread(src, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
        height, width = src.shape[:2]
        # downscale image first
        method = cv2.INTER_AREA  # cv2.INTER_AREA
        resolution = 64
        dummy = cv2.resize(src, (0,0),
                           fx=resolution/float(width),
                           fy=resolution/float(height),
                           interpolation=method)
        cv2.imwrite(dst, dummy)
        print("Texture %dx%d %s" % (resolution, resolution, dst))
    
            
def generate_from_grid(proj, group, ref_image=False, src_dir=".",
                       analysis_dir=".", resolution=512 ):
    # make the textures if needed
    make_textures_opencv(src_dir, analysis_dir, proj.image_list, resolution)
    
    for name in group:
        image = proj.findImageByName(name)
        if len(image.grid_list) == 0:
            continue

        root, ext = os.path.splitext(image.name)
        name = os.path.join( analysis_dir, "models", root + ".egg" )
        print("EGG file name:", name)

        f = open(name, "w")
        f.write("<CoordinateSystem> { Z-Up }\n\n")
        # f.write("<Texture> tex { \"" + image.name + ".JPG\" }\n\n")
        f.write("<Texture> tex { \"dummy.jpg\" }\n\n")
        f.write("<VertexPool> surface {\n")

        # this is contructed in a weird way, but we generate the 2d
        # iteration in the same order that the original grid_list was
        # constucted so it works.
        width, height = proj.cam.get_image_params()
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

def share_edge(label, uv1, uv2, h, w):
    if uv1[0] == uv2[0]:
        if uv1[0] == 0 or uv1[0] == w:
            return True
    elif uv1[1] == uv2[1]:
        if uv1[1] == 0 or uv1[1] == h:
            return True
    else:
        return False
    
# use law of cosines and edge point data to determine if a side of the
# triangle lies on the edge and also has a skinny edge angle.
def is_skinny(tri, pts, uvs, h, w, thresh=0.2):
    A = np.array(pts[tri[0]])
    B = np.array(pts[tri[1]])
    C = np.array(pts[tri[2]])
    uvA = uvs[tri[0]]
    uvB = uvs[tri[1]]
    uvC = uvs[tri[2]]
    va = A - C
    vb = C - B
    vc = B - A
    a2 = va[0]*va[0] + va[1]*va[1]
    b2 = vb[0]*vb[0] + vb[1]*vb[1]
    c2 = vc[0]*vc[0] + vc[1]*vc[1]
    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)
    print("pts:", A, B, C)
    print("sides:", a, b, c)
    tmp1 = np.clip((a2 + c2 - b2)/(2*a*c), -1.0, 1.0)
    tmp2 = np.clip((a2 + b2 - c2)/(2*a*b), -1.0, 1.0)
    alpha = math.acos(tmp1)
    beta = math.acos(tmp2)
    gamma = math.pi - (alpha + beta)
    print("angles:", alpha, beta, gamma)
    skinny = False
    if share_edge("c", uvA, uvB, h, w) and (alpha < thresh or gamma < thresh):
        skinny = True
    elif share_edge("b", uvB, uvC, h, w) and (gamma < thresh or beta < thresh):
        skinny = True
    elif share_edge("a", uvC, uvA, h, w) and (beta < thresh or alpha < thresh):
        skinny = True
    if skinny: print("skinny")
    return skinny
    
def generate_from_fit(proj, group, ref_image=False, src_dir=".",
                      analysis_dir=".", resolution=512 ):
    # make the textures if needed
    make_textures_opencv(src_dir, analysis_dir, proj.image_list, resolution)

    for name in group:
        image = proj.findImageByName(name)
        print("generate from fit:", image)
        if len(image.fit_xy) < 3:
            continue
            print("Warning: removing egg file, no polygons for:", name)
            if os.path.exists(name):
                os.remove(name)

        root, ext = os.path.splitext(image.name)
        name = os.path.join( analysis_dir, "models", root + ".egg" )
        print("EGG file name:", name)

        f = open(name, "w")
        f.write("<CoordinateSystem> { Z-Up }\n\n")
        f.write("<Texture> tex { \"" + image.name + ".JPG\" }\n\n")
        f.write("<VertexPool> surface {\n")

        width, height = proj.cam.get_image_params()
        n = 1
        #print("uv len:", len(image.fit_uv))
        for i in range(len(image.fit_xy)):
            f.write("  <Vertex> %d {\n" % n)
            f.write("    %.2f %.2f %.2f\n" % (image.fit_xy[i][0],
                                              image.fit_xy[i][1],
                                              image.fit_z[i]))
            u = np.clip(image.fit_uv[i][0]/float(width), 0, 1.0)
            v = np.clip(1.0-image.fit_uv[i][1]/float(height), 0, 1.0)
            f.write("    <UV> { %.5f %.5f }\n" % (u, v))
            f.write("  }\n")
            n += 1
        f.write("}\n\n")
        
        f.write("<Group> surface {\n")

        tris = scipy.spatial.Delaunay(np.array(image.fit_xy))
        for tri in tris.simplices:
            edge_count = image.fit_edge[tri[0]] \
                + image.fit_edge[tri[1]] \
                + image.fit_edge[tri[2]]
            if is_skinny(tri, image.fit_xy, image.fit_uv, height, width):
                print("skinny edge skipping")
            else:
                f.write("  <Polygon> {\n")
                f.write("   <TRef> { tex }\n")
                f.write("   <Normal> { 0 0 1 }\n")
                f.write("   <VertexRef> { %d %d %d <Ref> { surface } }\n" \
                        % (tri[0]+1, tri[1]+1, tri[2]+1))
                f.write("  }\n")
        f.write("}\n")
        f.close()
