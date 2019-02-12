#!/usr/bin/env python

'''
Feature-based image matching sample.

USAGE
  find_obj.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

import numpy as np
import cv2
from .common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(400)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img1, img2, kp_pairs, wscale=1.0, hscale=1.0, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis0 = vis.copy()

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    #p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    #p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    p1 = np.int32([kpp[0].pt for kpp in kp_pairs]) * (wscale, hscale)
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) * (wscale, hscale) + (w1, 0)

    p1 = np.int32(p1)
    p2 = np.int32(p2)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    def draw_keypoints(vis):
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)
            else:
                cv2.line(vis, (x1, y1), (x2, y2), red)
    draw_keypoints(vis)
    cv2.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        if event & cv2.EVENT_LBUTTONDOWN:
            vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            for i in idxs:
                status[i] = not status[i]
            draw_keypoints(vis)
            cv2.imshow(win, vis)

        # what is this supposed to do?  Currently it's broken.
        if event & cv2.EVENT_RBUTTONDOWN:
            vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                 (x1, y1), (x2, y2) = p1[i], p2[i]
                 col = (red, green)[status[i]]
                 cv2.line(vis, (x1, y1), (x2, y2), col)
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            vis = cv2.drawKeypoints(vis, kp1s, flags=4, color=kp_color)
            vis[:,w1:] = cv2.drawKeypoints(vis[:,w1:], kp2s, flags=4, color=kp_color)
            cv2.imshow(win, vis)

    cv2.setMouseCallback(win, onmouse)

    # keyboard input
    done = False
    while not done:
        print('waiting for keyboard input...')
        key = cv2.waitKey() & 0xff
        print("received %s (%d)" % (str(key), int(key)))
        if key == 27:
            # ESC = set all pairs to True (no delete) and exit
            for i in range(len(status)):
                status[i] = True
            done = True
        elif key == ord(' ') or key == ord('q'):
            # accept current selection and exit.  Different keystrokes
            # may have higher level meaning so we return the key as
            # well.
            done = True
        elif key == ord('y'):
            # set all pairs as valid
             for i in range(len(status)):
                status[i] = True
        elif key == ord('n'):
            # set all pairs as invalid
             for i in range(len(status)):
                status[i] = False
        # refresh display
        vis = vis0.copy()
        draw_keypoints(vis)
        cv2.imshow(win, vis)

    return key


if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try: fn1, fn2 = args
    except:
        fn1 = '../c/box.png'
        fn2 = '../c/box_in_scene.png'

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature_name)
    if detector != None:
        print('using', feature_name)
    else:
        print('unknown feature:', feature_name)
        sys.exit(1)


    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    def match_and_draw(win):
        print('matching...')
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))

        vis = explore_match(win, img1, img2, kp_pairs, status, H)

    match_and_draw('find_obj')
    cv2.waitKey()
    cv2.destroyAllWindows()
