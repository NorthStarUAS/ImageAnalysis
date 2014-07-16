import cv2
import math
import numpy as np

import ImageList

class Render():
    def __init__(self):
        self.image_list = []
        self.ref_lon = 0.0
        self.ref_lat = 0.0

    def setImageList(self, image_list):
        self.image_list = image_list

    def setRefCoord(self, lon, lat):
        self.ref_lon = lon
        self.ref_lat = lat

    def drawImage(self, image=None, source_dir=None,
                  cm_per_pixel=15.0, keypoints=False, bounds=None):
        if not len(image.corner_list):
            return
        if bounds == None:
            (xmin, ymin, xmax, ymax) = image.coverage()
        else:
            (xmin, ymin, xmax, ymax) = bounds
        x = int(100.0 * (xmax - xmin) / cm_per_pixel)
        y = int(100.0 * (ymax - ymin) / cm_per_pixel)
        print "Drawing %s: (%d %d)" % (image.name, x, y)
        #print str(image.corner_list)

        full_image = image.load_full_image(source_dir)
        h, w, d = full_image.shape
        corners = np.float32([[0,0],[w,0],[0,h],[w,h]])
        target = np.array([image.corner_list]).astype(np.float32)
        for i, pt in enumerate(target[0]):
            #print "i=%d" % i
            target[0][i][0] = 100.0 * (target[0][i][0] - xmin) / cm_per_pixel
            target[0][i][1] = 100.0 * (ymax - target[0][i][1]) / cm_per_pixel
        #print str(target)
        if keypoints:
            keypoints = []
            for i, kp in enumerate(image.kp_list):
                if image.kp_usage[i]:
                    keypoints.append(kp)
            src = cv2.drawKeypoints(full_image, keypoints,
                                    color=(0,255,0), flags=0)
        else:
            src = cv2.drawKeypoints(full_image, [],
                                    color=(0,255,0), flags=0)
        M = cv2.getPerspectiveTransform(corners, target)
        out = cv2.warpPerspective(src, M, (x,y))

        # clean up the edges so we don't have a ring of super dark pixels.
        ret, mask = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)
        kernel3 = np.ones((3,3),'uint8')
        mask = cv2.erode(mask, kernel3)
        out_clean = cv2.bitwise_and(out, mask)

        #cv2.imshow('output', out)
        #cv2.waitKey()
        return x, y, out_clean

    def compositeOverlayBottomup(self, base, new, blend_px=21):
        h, w, d = base.shape
        #print "h=%d w=%d d=%d" % ( h, w, d)

        # combine using masks and add operation (assumes pixel
        # image data will always be at least a little non-zero

        # create an inverse mask of the new region to be added
        newgray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)  
        ret, new_mask = cv2.threshold(newgray, 1, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('new_mask', new_mask)

        # dilate the mask (which shrinks the new area)
        blendsize = (blend_px,blend_px)
        kernel = np.ones(blendsize,'uint8')
        new_mask_dilate = cv2.dilate(new_mask, kernel)
        #cv2.imshow('new_mask_dilate', new_mask_dilate)

        # blur the mask to create a feathered edge
        new_mask_blur = cv2.blur(new_mask_dilate, blendsize)
        #cv2.imshow('new_mask_blur', new_mask_blur)

        # invert the blurred mask
        new_mask_blur_inv = 255 - new_mask_blur
        #cv2.imshow('new_mask_blur_inv', new_mask_blur_inv)

        new[:,:,0] = new[:,:,0] * (new_mask_blur_inv/255.0)
        new[:,:,1] = new[:,:,1] * (new_mask_blur_inv/255.0)
        new[:,:,2] = new[:,:,2] * (new_mask_blur_inv/255.0)
        #cv2.imshow('new masked', new)

        base[:,:,0] = base[:,:,0] * (new_mask_blur/255.0)
        base[:,:,1] = base[:,:,1] * (new_mask_blur/255.0)
        base[:,:,2] = base[:,:,2] * (new_mask_blur/255.0)
        #cv2.imshow('base masked', base)

        # And combine ...
        base = cv2.add(base, new)

        #cv2.imshow('base', base)
        #cv2.waitKey()

        return base
        
    def compositeOverlayTopdown(self, base, new, blend_px=21):
        h, w, d = base.shape
        #print "h=%d w=%d d=%d" % ( h, w, d)

        # combine using masks and add operation (assumes pixel
        # image data will always be at least a little non-zero

        # create an inverse mask of the current accumulated imagery
        basegray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)  
        ret, base_mask_inv = cv2.threshold(basegray, 1, 255,
                                           cv2.THRESH_BINARY_INV)
        #cv2.imshow('base_mask_inv', base_mask_inv)

        # create an inverse mask of the new region to be added
        newgray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)  
        ret, new_mask = cv2.threshold(newgray, 1, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('new_mask', new_mask)

        blendsize = (blend_px,blend_px)
        kernel = np.ones(blendsize,'uint8')
        base_mask_dilate = cv2.dilate(base_mask_inv, kernel)
        #cv2.imshow('base_mask_dilate', base_mask_dilate)
        base_mask_blur = cv2.blur(base_mask_dilate, blendsize)
        #cv2.imshow('base_mask_blur', base_mask_blur)

        base_mask_blur_inv = 255 - base_mask_blur
        #cv2.imshow('base_mask_blur_inv', base_mask_blur_inv)
        base_mask_blur_inv = base_mask_blur_inv | new_mask
        #cv2.imshow('base_mask_blur_inv2', base_mask_blur_inv)

        new[:,:,0] = new[:,:,0] * (base_mask_blur/255.0)
        new[:,:,1] = new[:,:,1] * (base_mask_blur/255.0)
        new[:,:,2] = new[:,:,2] * (base_mask_blur/255.0)
        #cv2.imshow('new masked', new)

        base[:,:,0] = base[:,:,0] * (base_mask_blur_inv/255.0)
        base[:,:,1] = base[:,:,1] * (base_mask_blur_inv/255.0)
        base[:,:,2] = base[:,:,2] * (base_mask_blur_inv/255.0)
        #cv2.imshow('base masked', base)

        fast = True
        if fast:
            # Now clip the new imagery against the area already covered
            #new = cv2.add(base, new, mask=mask_inv)

            # And combine ...
            base = cv2.add(base, new)

        else:
            # alpha blend using the mask as the alpha value, works but
            # is done the hardway because I can't find a native opencv
            # way to do this.
            mask_blur = cv2.blur(mask_inv, (50,50))
            for i in xrange(h):
                for j in xrange(w):
                    #(r0, g0, b0) = base[i][j]
                    #(r1, g1, b1) = new[i][j]
                    #a = mask_blur[i][j] / 255.0 
                    #r = r0*(1.0-a) + r1*a
                    #g = g0*(1.0-a) + g1*a
                    #b = b0*(1.0-a) + b1*a
                    #base = (r, g, b)
                    b = base[i][j]
                    n = new[i][j]
                    a = mask_blur[i][j] / 255.0
                    if n[0] + n[1] + n[2] > 0:
                        base[i][j][0] = b[0]*(1.0-a) + n[0]*a
                        base[i][j][1] = b[1]*(1.0-a) + n[1]*a
                        base[i][j][2] = b[2]*(1.0-a) + n[2]*a

        #cv2.imshow('base', base)
        #cv2.waitKey()

        return base
        
    def drawImages(self, draw_list=[], source_dir=None,
                   cm_per_pixel=15.0, blend_cm=200,
                   bounds=None, file=None, keypoints=False):
        print "drawImages() bounds = %s" % str(bounds)
        # compute blend diameter in consistent pixel units
        blend_px = int(blend_cm/cm_per_pixel)+1
        if blend_px % 2 == 0:
            blend_px += 1

        if bounds == None:
            (xmin, ymin, xmax, ymax) = ImageList.coverage(draw_list)
        else:
            (xmin, ymin, xmax, ymax) = bounds

        x = int(100.0 * (xmax - xmin) / cm_per_pixel)
        y = int(100.0 * (ymax - ymin) / cm_per_pixel)
        print "New image dimensions: (%d %d)" % (x, y)
        base_image = np.zeros((y,x,3), np.uint8)

        for image in reversed(draw_list):
            w, h, out = self.drawImage(image, source_dir,
                                       cm_per_pixel,
                                       keypoints,
                                       bounds=(xmin, ymin, xmax, ymax))
            base_image = self.compositeOverlayBottomup(base_image, out,
                                                       blend_px)
            #cv2.imshow('output', base_image)
            #cv2.waitKey()

        cv2.imwrite(file, base_image)

    def drawSquare(self, placed_list, source_dir=None,
                   cm_per_pixel=15.0, blend_cm=200, bounds=None, file=None):
        (xmin, ymin, xmax, ymax) = bounds
        xcenter = (xmin + xmax) * 0.5
        ycenter = (ymin + ymax) * 0.5
        pad = (xmax - xmin) * 0.5
        draw_list = ImageList.getImagesCoveringPoint(placed_list, 
                                                     xcenter, ycenter, pad,
                                                     only_placed=True)
        if len(draw_list):
            self.drawImages( draw_list, source_dir=source_dir,
                             cm_per_pixel=cm_per_pixel, blend_cm=blend_cm,
                             bounds=bounds, file=file)
        return draw_list

    def x2lon(self, x):
        nm2m = 1852.0
        x_nm = x / nm2m
        factor = math.cos(self.ref_lat*math.pi/180.0)
        x_deg = (x_nm / 60.0) / factor
        return x_deg + self.ref_lon

    def y2lat(self, y):
        nm2m = 1852.0
        y_nm = y / nm2m
        y_deg = y_nm / 60.0
        return y_deg + self.ref_lat
        
        
    def drawGrid(self, placed_list, source_dir=None,
                 cm_per_pixel=15.0, blend_cm=200,
                 dim=4096):
        # compute blend diameter in consistent pixel units
        blend_px = int(blend_cm/cm_per_pixel)+1
        if blend_px % 2 == 0:
            blend_px += 1

        (xmin, ymin, xmax, ymax) = ImageList.coverage(self.image_list)
        grid_m = (dim * cm_per_pixel) / 100.0
        print "grid square size = (%.2f x %.2f)" % (grid_m, grid_m)
        #xpixel = (xmax - xmin) * 100.0 / cm_per_pixel
        #ypixel = (ymax - ymin) * 100.0 / cm_per_pixel

        f = open('gdalscript.sh', 'w')
        f.write('#!/bin/sh\n\n')
        f.write('rm -f tile*.tif\n')
        count = 0
        y = ymin
        while y < ymax:
            x = xmin
            while x < xmax:
                print "grid = (%.2f %.2f)" % (x, y)
                base = "tile%03d" % count
                jpgfile = base + ".jpg"
                tifffile = base + ".tif"
                images = self.drawSquare( placed_list, source_dir=source_dir,
                                          cm_per_pixel=cm_per_pixel,
                                          blend_cm=blend_cm,
                                          bounds=(x, y, x+grid_m, y+grid_m),
                                          file=jpgfile)
                if len(images):
                    (ul_lon, ul_lat) = ImageList.cart2wgs84(x, y+grid_m,
                                                            self.ref_lon,
                                                            self.ref_lat)
                    (lr_lon, lr_lat) = ImageList.cart2wgs84(x+grid_m, y,
                                                            self.ref_lon,
                                                            self.ref_lat)
                    cmd = 'gdal_translate -a_srs "+proj=latlong +datum=WGS84" '
                    cmd += '-of GTiff -co "INTERLEAVE=PIXEL" '
                    cmd += '-a_ullr %.15f %.15f %.15f %.15f ' % \
                           ( ul_lon, ul_lat, lr_lon, lr_lat )
                    cmd += '%s %s\n' % (jpgfile, tifffile)
                    f.write('echo running gdal_translate...\n')
                    f.write(cmd)
                    count += 1
                x += grid_m
            y += grid_m
        f.write('rm output.tif\n')
        f.write('echo running gdal_merge\n')
        f.write('gdal_merge.py -o output.tif tile*.tif\n')
        f.write('echo running gdalwarp\n')
        f.write('rm output_3857.tif\n')
        f.write('gdalwarp -t_srs EPSG:3857 output.tif output_3857.tif\n')
        f.write('echo running gdal2tiles.py\n')
        f.write('rm -rf output\n')
        f.write('gdal2tiles.py -z 16-21 -s_srs=EPSG:3857 output_3857.tif output\n')
        f.close()
