import cv2
import numpy as np

class Render():
    def __init__(self):
        self.image_list = []

    def setImageList(self, image_list):
        self.image_list = image_list

    def imageCoverage(self, image):
        if not len(image.corner_list):
            return (0.0, 0.0, 0.0, 0.0)

        # find the min/max area of the image
        p0 = image.corner_list[0]
        minx = p0[0]; maxx = p0[0]; miny = p0[1]; maxy = p0[1]
        for pt in image.corner_list:
            if pt[0] < minx:
                minx = pt[0]
            if pt[0] > maxx:
                maxx = pt[0]
            if pt[1] < miny:
                miny = pt[1]
            if pt[1] > maxy:
                maxy = pt[1]
        print "%s coverage: (%.2f %.2f) (%.2f %.2f)" \
            % (image.name, minx, miny, maxx, maxy)
        return (minx, miny, maxx, maxy)

    # return a list of images that cover the given point within 'pad'
    # or are within 'pad' distance of touching the point.
    def getImagesCoveringPoint(self, x=0.0, y=0.0, pad=20.0):
        # build list of images covering target point
        coverage_list = []
        for image in self.image_list:
            (x0, y0, x1, y1) = self.imageCoverage(image)
            if x >= x0-pad and x <= x1+pad:
                if y >= y0-pad and y <= y1+pad:
                    if image.connections > 0:
                        # only add images that connect to other images
                        coverage_list.append(image)

        # sort by # of connections
        print "presort = %s" % str(coverage_list)
        coverage_list = sorted(coverage_list,
                               key=lambda image: image.connections,
                               reverse=True)
        print "postsort = %s" % str(coverage_list)
        return coverage_list

    def drawImage(self, image=None, source_dir=None,
                  cm_per_pixel=15.0, keypoints=False, bounds=None):
        if not len(image.corner_list):
            return
        if bounds == None:
            (minx, miny, maxx, maxy) = self.imageCoverage(image)
        else:
            (minx, miny, maxx, maxy) = bounds
        x = int(100.0 * (maxx - minx) / cm_per_pixel)
        y = int(100.0 * (maxy - miny) / cm_per_pixel)
        print "Drawing %s: (%d %d)" % (image.name, x, y)
        #print str(image.corner_list)

        full_image = image.load_full_image(source_dir)
        h, w, d = full_image.shape
        corners = np.float32([[0,0],[w,0],[0,h],[w,h]])
        target = np.array([image.corner_list]).astype(np.float32)
        for i, pt in enumerate(target[0]):
            #print "i=%d" % i
            target[0][i][0] = 100.0 * (target[0][i][0] - minx) / cm_per_pixel
            target[0][i][1] = 100.0 * (maxy - target[0][i][1]) / cm_per_pixel
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

    def compositeOverlay(self, base, new, blend_px=21):
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
                   keypoints=False):
        # compute blend diameter in consistent pixel units
        blend_px = int(blend_cm/cm_per_pixel)+1
        if blend_px % 2 == 0:
            blend_px += 1

        minx = None; maxx = None; miny = None; maxy = None
        for image in draw_list:
            (x0, y0, x1, y1) = self.imageCoverage(image)
            if minx == None or x0 < minx:
                minx = x0
            if miny == None or y0 < miny:
                miny = y0
            if maxx == None or x1 > maxx:
                maxx = x1
            if maxy == None or y1 > maxy:
                maxy = y1
        print "Group area coverage: (%.2f %.2f) (%.2f %.2f)" \
            % (minx, miny, maxx, maxy)

        x = int(100.0 * (maxx - minx) / cm_per_pixel)
        y = int(100.0 * (maxy - miny) / cm_per_pixel)
        print "New image dimensions: (%d %d)" % (x, y)
        base_image = np.zeros((y,x,3), np.uint8)

        for image in draw_list:
            w, h, out = self.drawImage(image, source_dir,
                                       cm_per_pixel,
                                       keypoints,
                                       bounds=(minx, miny, maxx, maxy))
            base_image = self.compositeOverlay(base_image, out, blend_px)
            #(x0, y0, x1, y1) = self.imageCoverage(image)
            #w0 = int(100.0 * (x0 - minx) / cm_per_pixel)
            #h0 = int(100.0 * (maxy - y1) / cm_per_pixel)
            #print "roi (%d:%d %d:%d)" % ( w0, w, h0  , h )
            #roi = blank_image[h0:h, w0:w]
            #roi = out
            #roi = np.ones((h,w,3), np.uint8)

            #cv2.imshow('output', base_image)
            #cv2.waitKey()
        output_name = "output.jpg"
        cv2.imwrite(output_name, base_image)

        #s_img = cv2.imread("smaller_image.png", -1)
        #for c in range(0,3):
        #    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] = s_img[:,:,c] * (s_img[:,:,3]/255.0) +  l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)

