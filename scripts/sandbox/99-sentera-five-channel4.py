#!/usr/bin/python3

import argparse
import cv2
import math
import numpy as np
import os
import pyexiv2                  # dnf install python3-exiv2 (py3exiv2)
from tqdm import tqdm
import matplotlib.pyplot as plt

from props import root, getNode
import props_json

from lib import Camera
from lib import Image

parser = argparse.ArgumentParser(description='Align and combine sentera images.')
parser.add_argument('--flight', help='Base directory (parent of NIR/RGB directories.')
parser.add_argument('--scale', type=float, default=0.4, help='scale image before processing')
parser.add_argument('--image', help='image name')
parser.add_argument('image1', help='image1 path')
parser.add_argument('image2', help='image1 path')
args = parser.parse_args()

def detect_camera(image_path):
    camera = ""
    exif = pyexiv2.ImageMetadata(image_path)
    exif.read()
    if 'Exif.Image.Make' in exif:
        camera = exif['Exif.Image.Make'].value
    if 'Exif.Image.Model' in exif:
        camera += '_' + exif['Exif.Image.Model'].value
    if 'Exif.Photo.LensModel' in exif:
        camera += '_' + exif['Exif.Photo.LensModel'].value
    camera = camera.replace(' ', '_')
    return camera

if args.flight and args.image:
    image1 = os.path.join(args.flight, 'NIR', args.image)
    image2 = os.path.join(args.flight, 'RGB', args.image)
else:
    image1 = args.image1
    image2 = args.image2
    
cam1 = detect_camera(image1)
cam2 = detect_camera(image2)
print(cam1)
print(cam2)

cam1_node = getNode("/camera1", True)
cam2_node = getNode("/camera2", True)

if props_json.load(os.path.join("../cameras", cam1 + ".json"), cam1_node):
    print("successfully loaded cam1 config")
if props_json.load(os.path.join("../cameras", cam2 + ".json"), cam2_node):
    print("successfully loaded cam2 config")

tmp = []
for i in range(9):
    tmp.append( cam1_node.getFloatEnum('K', i) )
K1 = np.copy(np.array(tmp)).reshape(3,3)
print("K1:", K1)

tmp = []
for i in range(5):
    tmp.append( cam1_node.getFloatEnum('dist_coeffs', i) )
dist1 = np.array(tmp)
print("dist1:", dist1)

tmp = []
for i in range(9):
    tmp.append( cam2_node.getFloatEnum('K', i) )
K2 = np.copy(np.array(tmp)).reshape(3,3)
print("K2:", K2)

tmp = []
for i in range(5):
    tmp.append( cam2_node.getFloatEnum('dist_coeffs', i) )
dist2 = np.array(tmp)
print("dist2:", dist2)

i1 = cv2.imread(image1, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
i2 = cv2.imread(image2, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
if i1 is None:
    print("Error loading:", image1)
    quit()
if i2 is None:
    print("Error loading:", image2)
    quit()

i1 = cv2.undistort(i1, K1, dist1)
i2 = cv2.undistort(i2, K1, dist1)

# scale images (anticipating images are identical dimensions, but this
# will force that assumption if the happen to not be.)
(h, w) = i1.shape[:2]
i1 = cv2.resize(i1, (int(w*args.scale), int(h*args.scale)))
i2 = cv2.resize(i2, (int(w*args.scale), int(h*args.scale)))

detector = cv2.xfeatures2d.SIFT_create()
kp1 = detector.detect(i1)
kp2 = detector.detect(i2)
print("Keypoints:", len(kp1), len(kp2))

kp1, des1 = detector.compute(i1, kp1)
kp2, des2 = detector.compute(i2, kp2)
print("Descriptors:", len(des1), len(des2))

FLANN_INDEX_KDTREE = 1
flann_params = {
    'algorithm': FLANN_INDEX_KDTREE,
    'trees': 5
}
search_params = dict(checks=100)
matcher = cv2.FlannBasedMatcher(flann_params, search_params)
matches = matcher.knnMatch(des1, des2, k=5)
print("Raw matches:", len(matches))

if False:
    plt.figure()
    plt.title('match distance fall off')
    for i, m in enumerate(tqdm(matches)):
        if i % 10 == 0:
            for j in m:
                vals = []
                pos = []
                for j in range(len(m)):
                    pos.append(j)
                    vals.append(m[j].distance)
            plt.plot(pos, vals, lw=1)
    plt.show()

def draw_inlier(src1, src2, kpt1, kpt2, inlier, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == 'ONLY_LINES':
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1
.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 
255, 255))

    elif drawing_type == 'LINES_AND_POINTS':
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))
        for i in range(len(inlier)):
            left = kpt1[inlier[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kpt2[inlier[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    cv2.imshow('show', output)
    cv2.waitKey()

def decomposeAffine(affine):
        tx = affine[0][2]
        ty = affine[1][2]

        a = affine[0][0]
        b = affine[0][1]
        c = affine[1][0]
        d = affine[1][1]

        sx = math.sqrt( a*a + b*b )
        if a < 0.0:
            sx = -sx
        sy = math.sqrt( c*c + d*d )
        if d < 0.0:
            sy = -sy

        rotate_deg = math.atan2(-b,a) * 180.0/math.pi
        if rotate_deg < -180.0:
            rotate_deg += 360.0
        if rotate_deg > 180.0:
            rotate_deg -= 360.0
        return (rotate_deg, tx, ty, sx, sy)

def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)    

def ndre_helper(lut, a, b, c1, c2):
    print('range:', a, b)
    db = c2[0] - c1[0]
    dg = c2[1] - c1[1]
    dr = c2[2] - c1[2]
    for i in range(a, b + 1):
        percent = (i - a) / (b - a)
        lut[0][i][0] = int(c1[0] + db * percent)
        lut[0][i][1] = int(c1[1] + dg * percent)
        lut[0][i][2] = int(c1[2] + dr * percent)
        print(' ', i, percent, lut[0][i])
        
def make_lut_ndre(cutoffs, colors):
    lut = np.zeros( (1, 256, 3) ).astype('uint8')
    ndre_helper(lut, 0, cutoffs[0], colors[0], colors[0])
    size = len(cutoffs)
    for i in range(0, size-1):
        ndre_helper(lut, cutoffs[i], cutoffs[i+1], colors[i], colors[i+1])
    ndre_helper(lut, cutoffs[size-1], 255, colors[size-1], colors[size-1])
    return lut

def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img_norm = (img - min) / (max - min)
    return img_norm
    
angle_bins = [0] * 91
dist_bins = [0] * (int(math.sqrt(w*w+h*h)/10.0)+1)

H = np.identity(3);
first_iteration = True
while True:
    print('H:', H)
    src_pts = np.float32([kp1[i].pt for i in range(len(kp1))]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[i].pt for i in range(len(kp2))]).reshape(-1, 1, 2)
    src_pts = cv2.perspectiveTransform(src_pts, H)
    #print('src:', src_pts)
    #print('dst:', dst_pts)
    
    print("collect stats...")
    match_stats = []
    for i, m in enumerate(tqdm(matches)):
        best_index = 0
        best_value = 99999999999999999999999.9
        best_angle = 0
        best_size = 0
        best_dist = 0
        ratio = m[0].distance / m[1].distance
        for j in range(len(m)):
            p1 = src_pts[m[j].queryIdx]
            p2 = dst_pts[m[j].trainIdx]
            #print(p1, p2)
            raw_dist = np.linalg.norm(p1-p2)
            if first_iteration:
                # first iteration don't use distance
                px_dist = 1
            else: 
                px_dist = 1 + raw_dist*raw_dist
            a1 = np.array(kp1[m[j].queryIdx].angle)
            a2 = np.array(kp2[m[j].trainIdx].angle)
            # angle difference mapped to +/- 180
            # angle = (a1-a2+180) % 360 - 180
            # angle difference mapped to +/- 90
            angle = (a1-a2+90) % 180 - 90
            #angle = 1
            #print(a1, a2, angle)
            angle_dist = abs(angle) + 1
            s1 = np.array(kp1[m[j].queryIdx].size)
            s2 = np.array(kp2[m[j].trainIdx].size)
            size_diff = abs(s1 - s2) + 1
            metric = (px_dist * angle_dist * size_diff) / ratio
            #print(" ", j, m[j].distance, px_dist, abs(1 + angle), size_diff, metric)
            if metric < best_value:
                best_value = metric
                best_index = j
                best_angle = abs(angle)
                best_size = size_diff
                best_dist = raw_dist
        #print(i, best_index, m[best_index].distance, best_angle, best_size, best_value)
        match_stats.append( [ m[best_index], ratio, best_value, best_angle,
                              best_size, best_dist ] )
        dist_bins[int(best_dist/10.0)] += 1
        angle_bins[int(round(abs(best_angle)))] += 1

    if first_iteration:
        target_dist = np.argmax(dist_bins)*10
    else:
        target_dist = 0.0
    print("target dist:", target_dist)
    
    target_angle = np.argmax(angle_bins)
    print("target angle:", target_angle)

    # select the best subset of matches
    filt_matches = []
    for i, line in enumerate(tqdm(match_stats)):
        match = line[0]
        ratio = line[1]
        best_value = line[2]
        best_angle = line[3]
        best_size = line[4]
        best_dist = line[5]
        if ratio < 0.60:
            # passes ratio test as per Lowe's paper
            filt_matches.append(match)
        else:
            if abs(best_dist - target_dist) > 30:
                continue
            if abs(best_angle - target_angle) > 5:
                continue
            elif best_size > 2:
                continue
            elif (first_iteration and best_value < 5) or (not first_iteration and best_value < 500):
                print(i, best_index, match.distance, best_angle, best_size, best_value)

                filt_matches.append(match)
    print("Filtered matches:", len(filt_matches))
    first_iteration = False

    if False:
        # dist histogram
        plt.figure()
        y_pos = np.arange(len(dist_bins))
        plt.bar(y_pos, dist_bins, align='center', alpha=0.5)
        plt.xticks(y_pos, range(len(dist_bins)))
        plt.ylabel('count')
        plt.title('total distance histogram')

        # angle histogram
        plt.figure()
        y_pos = np.arange(len(angle_bins))
        plt.bar(y_pos, angle_bins, align='center', alpha=0.5)
        plt.xticks(y_pos, range(len(angle_bins)))
        plt.ylabel('count')
        plt.title('angle histogram')

        plt.show()

    if False:
        sh1 = i1.shape
        sh2 = i2.shape
        size1 = (sh1[1], sh1[0])
        size2 = (sh2[1], sh2[0])
        matchesGMS = cv2.xfeatures2d.matchGMS(size1, size2, kp1, kp2, filt_matches, withRotation=False, withScale=False, thresholdFactor=5.0)
        print("GMS matches:", len(matchesGMS))
    
    if True:
        print("Filtering by findHomography")
        tol = 4.0
        src = []
        dst = []
        for m in filt_matches:
            src.append( kp1[m.queryIdx].pt )
            dst.append( kp2[m.trainIdx].pt )
        H, status = cv2.findHomography(np.array([src]).astype(np.float32),
                                       np.array([dst]).astype(np.float32),
                                       cv2.RANSAC,
                                       tol)
        matches_fit = []
        for i, m in enumerate(filt_matches):
            if status[i]:
                matches_fit.append(m)

    print("Fitted matches:", len(matches_fit))
    draw_inlier(i1, i2, kp1, kp2, matches_fit, 'ONLY_LINES')

    if True:
        src = []
        dst = []
        for m in matches_fit:
            src.append( kp1[m.queryIdx].pt )
            dst.append( kp2[m.trainIdx].pt )
        affine, status = \
            cv2.estimateAffinePartial2D(np.array([src]).astype(np.float32),
                                        np.array([dst]).astype(np.float32))
        (rot, tx, ty, sx, sy) = decomposeAffine(affine)
        print("Affine:")
        print("Rotation (deg):", rot)
        print("Translation (pixels):", tx, ty)
        print("Skew:", sx, sy)
        # H, status = cv2.findHomography(np.array([src]).astype(np.float32),
        #                                         np.array([dst]).astype(np.float32),
        #                                         cv2.LMEDS)

    print("Homography:", H)
    # (rot, tx, ty, sx, sy) = decomposeAffine(affine)
    # print("Affine:")
    # print("Rotation (deg):", rot)
    # print("Translation (pixels):", tx, ty)
    # print("Skew:", sx, sy)


    #i1_new = cv2.warpAffine(i1, affine, (i1.shape[1], i1.shape[0]))
    i1_new = cv2.warpPerspective(i1, H, (i1.shape[1], i1.shape[0]))
    blend = cv2.addWeighted(i1_new, 0.5, i2, 0.5, 0)

    cv2.imshow('i1', i1)
    cv2.imshow('i1_new', i1_new)
    cv2.imshow('i2', i2)
    cv2.imshow('blend', blend)

    if True:
        # NDRE test
        ndvi_lut = cv2.imread("NDVI_Scale_LDP.png", flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
        ndvi_lut = np.reshape(ndvi_lut, (1, 256, 3))
        print(ndvi_lut.shape, ndvi_lut.dtype)
        #print(make_lut_ndre().shape, make_lut_ndre().dtype)
        nir, garbage, re = cv2.split(i1_new)
        g, b, r = cv2.split(i2)
        cv2.imshow('ndre nir', nir)
        #cv2.imshow('ndre garbage', garbage)
        cv2.imshow('ndre re', re)
        cutoffs = []
        if False:
            # sentera formala for ndre
            #((-0.341*nir_red + 2.426*nir_blue)- (1.0*nir_red - 0.956*nir_blue))/ ( (-0.341*nir_red + 2.426*nir_blue)+(1.0*nir_red - 0.956*nir_blue))
            nnir = normalize(-0.341*re + 2.426*nir)
            nnir[nnir==0] = 1
            nre = normalize(1.0*re - 0.956*nir)
            nindex = (nnir - nre) / (nnir + nre)
            cutoffs = [ 32, 60, 80, 110, 168 ]
            print('nir', nnir.shape, np.min(nnir), np.max(nnir))
            print('re', nre.shape, np.min(nre), np.max(nre))
            print('ndvi', nindex.shape, np.min(nindex), np.max(nindex))
        elif False:
            # using the inversion of re as a proxy for red (modified NDRE)
            nnir = nir/255.0
            nre = 1.0 - re/255.0
            nindex = (nnir - nre) / (nnir + nre)
            cutoffs = [ 32, 60, 80, 110, 168 ]
            print('nir', nnir.shape, np.min(nnir), np.max(nnir))
            print('re', nre.shape, np.min(nre), np.max(nre))
            print('ndvi', nindex.shape, np.min(nindex), np.max(nindex))
        elif True:
            # using nir and r for traditional ndvi
            nir[nir==0] = 1
            nnir = nir/255.0
            nre = r/255.0
            nindex = (nnir - nre) / (nnir + nre)
            cutoffs = [ 42, 70, 90, 110, 168 ]
            print('nir', nnir.shape, np.min(nnir), np.max(nnir))
            print('re', nre.shape, np.min(nre), np.max(nre))
            print('ndvi', nindex.shape, np.min(nindex), np.max(nindex))
        elif False:
            print("RE + 1-R")
            # using nir and 1-r to highlight healthy plants with red
            vnir = re/255.0
            g[g==0] = 1
            vre = 0.5 * (g/r)
            vre[vre>1] = 1
            cutoffs = [ 48, 60, 88, 128, 200 ]   # 0.5
            # cutoffs = [ 0, 32, 42, 64, 96, 255 ]   # 0.3
            # cutoffs = [ 0, 11, 15, 22, 40, 255 ]   # 0.1
        index = ((0.5 * nindex + 0.5) * 255).astype('uint8')

        colors = [
            (0x17, 0x36, 0xff),
            (0x00, 0x7b, 0xff),
            (0x0b, 0xde, 0xff),
            (0x60, 0xff, 0xdf),
            (0x3f, 0xff, 0x6f),
            (0x1e, 0xb8, 0x30)
        ]
        
        hist,bins = np.histogram(index.flatten(),256,[0,256])
        #hist[255] = 0
        cdf = hist[1:].cumsum()
        print('cdf:', cdf, cdf.size)
        if False:
            # equal area histogram coloring for NDRE
            cutoffs = [0]
            step = cdf.max() / 5
            for i in range(0, cdf.size):
                print(i, cdf[i])
                if cdf[i] > len(cutoffs) * step:
                    cutoffs.append(i)
            cutoffs.append(255)
        print(cutoffs)
            
        ndre_color = cv2.cvtColor(index, cv2.COLOR_GRAY2BGR)
        ndre_mapped = cv2.LUT(ndre_color, make_lut_ndre(cutoffs, colors))
        cv2.imshow('normalized index', index)
        cv2.imshow('index color mapped', ndre_mapped)
        cdf_normalized = cdf * hist.max()/ cdf.max()

        # Blob test
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 127
        params.maxThreshold = 255
        params.thresholdStep = 128
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 25
        params.maxArea = 200
        # Filter by Circularity
        #params.filterByCircularity = True
        #params.minCircularity = 0.1
        # Filter by Convexity
        #params.filterByConvexity = True
        #params.minConvexity = 0.87
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.0
        params.maxInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        index_inv = 255 - index
        keypoints = detector.detect(index_inv)
        cv2.imshow('index inverted', index_inv)
        
        img2 = ndre_color
        for marker in keypoints:
            m = tuple(int(i) for i in marker.pt)
            print('blob:', m)
            img2 = cv2.circle(img2, m, int(marker.size), (0, 238, 0), 1, cv2.LINE_AA)
        #img2 = cv2.drawMarker(img2, m, color=(0, 255, 0))
        cv2.imshow("Blobs", img2)

        cv2.waitKey()
        
        plt.plot(cdf * hist.max() / cdf.max(), color = 'b')
        plt.hist(index.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.show()

    if False:
        # RGB test
        b, g, r = cv2.split(i2)
        cv2.imshow('rgb b', b)
        cv2.imshow('rgb g', g)
        cv2.imshow('rgb r', r)

    if False:
        # YUV test
        img_yuv = cv2.cvtColor(i2, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        lut_u, lut_v = make_lut_u(), make_lut_v()
        # Convert back to BGR so we can apply the LUT
        u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        u_mapped = cv2.LUT(u, lut_u)
        v_mapped = cv2.LUT(v, lut_v)
        cv2.imshow('u_mapped', u_mapped)
        cv2.imshow('v_mapped', v_mapped)

    if False:
        # HSV test
        img_hsv = cv2.cvtColor(i2, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        lut_u, lut_v = make_lut_u(), make_lut_v()
        # Convert back to BGR so we can apply the LUT
        h = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        h_mapped = cv2.LUT(h, lut_u)
        s_mapped = cv2.LUT(s, lut_v)
        cv2.imshow('h_mapped', h_mapped)
        cv2.imshow('s_mapped', s_mapped)

if False:
    i1g = cv2.cvtColor(i1_new, cv2.COLOR_BGR2GRAY)
    i2g = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    print(i1g.dtype, i2g.dtype)
    print(i1.shape, i2.shape)
    print(i1g.shape, i2g.shape)

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=31)
    disparity = stereo.compute(i1g, i2g)
    print(disparity.dtype, disparity.shape, np.amax(disparity))
    scaled = disparity.astype('float64')*256/np.amax(disparity)
    cv2.imshow('disparity', scaled.astype('uint8'))
    cv2.waitKey()

    flow = cv2.calcOpticalFlowFarneback(i1g, i2g, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(i1)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    cv2.waitKey()
