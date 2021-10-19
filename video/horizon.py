import cv2
import math
import numpy as np

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

# a, b are line end points, p is some other point
# returns the closest point on ab to p (orthogonal projection)
def ClosestPointOnLine(a, b, p):
    ap = p - a
    ab = b - a
    return a + np.dot(ap,ab) / np.dot(ab,ab) * ab

# locate horizon and estimate relative roll/pitch of the camera,
# returns hough lines found (sorted by most dominant first.)
def horizon(frame, do_otsu=True):
    # attempt to threshold on high blue values (blue<->white)
    b, g, r = cv2.split(frame)
    cv2.imshow("b", b)
    #cv2.imshow("g", g)
    #cv2.imshow("r", r)
    #print("shape:", frame.shape)
    #print("blue range:", np.min(b), np.max(b))
    #print('ave:', np.average(b), np.average(g), np.average(r))

    # the lower the 1st canny number the more total edges are accepted
    # the lower the 2nd canny number the less hard the edges need to
    # be to accept an edge
    #edges = cv2.Canny(b, 200, 600)
    if do_otsu:
        edges = cv2.Canny(b, 50, 150)
    else:
        edges = cv2.Canny(b, 25, 75)
    cv2.imshow("edges", edges)

    if do_otsu:
        # Otsu thresholding on blue channel
        ret2, otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #print('ret2:', ret2)
        cv2.imshow('otsu mask', otsu)

        # Do a simple structural analysis and find the largest/upper-most
        # connected area that is part of the sky and use that as our sky
        # mask
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu)
        #print("retval:", retval)
        #print("labels:", labels.shape, labels)
        #print("stats:", stats)
        max_metric = 0
        max_index = -1
        max_mask = None
        h,w = b.shape
        for i in range(retval):
            top = stats[i,1]
            height = stats[i,3]
            pos = top + height*0.5
            pixels = stats[i,4]
            metric = pixels * (h-pos)/h
            #print("metric:", pixels, top, metric)
            if metric > max_metric:
                lmask = np.uint8(labels==i)*255
                mask = cv2.bitwise_and(otsu, otsu, mask=lmask)
                if np.any(mask):
                    max_metric = metric
                    max_index = i
                    max_mask = mask.copy()
        cv2.imshow('max mask', max_mask.astype(np.uint8))
        #print("centroids:", centroids)

        # dilate the mask a small bit
        kernel = np.ones((5,5), np.uint8)
        max_mask = cv2.dilate(max_mask, kernel, iterations=1)
  
        # global thresholding on blue channel before edge detection
        #thresh = cv2.inRange(frame, (210, 0, 0), (255, 255, 255))
        #cv2.imshow('global mask', thresh)

        preview = cv2.bitwise_and(frame, frame, mask=max_mask)
        cv2.imshow("threshold", preview)

        # Use the blue mask (Otsu) to filter out edge noise in area we
        # don't care about (to improve performance of the hough transform)
        edges = cv2.bitwise_and(edges, edges, mask=max_mask)
        cv2.imshow("masked edges", edges)
    
    #theta_res = np.pi/180       # 1 degree
    theta_res = np.pi/1800      # 0.1 degree
    threshold = int(frame.shape[1] / 8)

    # Standard hough transform.  Presumably returns lines sorted
    # by most dominant first
    lines = cv2.HoughLines(edges, 1, theta_res, threshold)
    return lines

# track best line (possibly this has errors, or it is exteremly
# non-productive) Not used, but left here in case I want to revive the
# concept and debug.
best_rho = 0
best_theta = 0
def track_best(lines):
    global best_rho
    global best_theta
    bi = -1
    bm = 0
    for i, line in enumerate(lines[:2]): # just top two lines
        rho, theta = line[0]
        rd = best_rho - rho
        td = best_theta - theta
        metric = rd*rd + td*td
        print(i, metric)
        if bi < 0 or metric < bm:
            bi = i
            bm = metric
    print("best:", bi)
    best_rho = lines[bi][0][0]
    best_theta = lines[bi][0][1]
    return lines[bi]
        
# get the roll/pitch of camera orientation relative to specified
# horizon line
def get_camera_attitude(line, IK, cu, cv):
    # print('line:', line)
    rho, theta = line[0]
    #print("theta:", theta * r2d)
    roll = 90 - theta*r2d
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    #print("p0:", x0, y0)
    len2 = 1000
    x1 = int(x0 + len2*(-b))
    y1 = int(y0 + len2*(a))
    x2 = int(x0 - len2*(-b))
    y2 = int(y0 - len2*(a))

    if False:
        # temp test
        w = cu * 2
        h = cv * 2
        for p in [ (0, 0, 1), (w, 0, 1), (0, h, 1), (w, h, 1), (cu, cv, 1) ]:
            uvh = np.array(p)
            proj = IK.dot(uvh)
            print(p, "->", proj)
        
    p0 = ClosestPointOnLine(np.array([x1,y1]),
                            np.array([x2,y2]),
                            np.array([cu,cv]))
    uvh = np.array([p0[0], p0[1], 1.0])
    proj = IK.dot(uvh)
    #print("proj:", proj, proj/np.linalg.norm(proj))
    dot_product = np.dot(np.array([0,0,1]), proj/np.linalg.norm(proj))
    pitch = np.arccos(dot_product) * r2d
    if p0[1] < cv:
        pitch = -pitch
    #print("roll: %.1f pitch: %.1f" % (roll, pitch))
    return roll, pitch

# line is the first array entry from cv2.HoughLines()
def draw(frame, line, IK, cu, cv):
    #print('line:', line)
    rho, theta = line[0]
    #print("theta:", theta * r2d)
    roll = 90 - theta*r2d
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    #print("p0:", x0, y0)
    len2 = 1000
    x1 = int(x0 + len2*(-b))
    y1 = int(y0 + len2*(a))
    x2 = int(x0 - len2*(-b))
    y2 = int(y0 - len2*(a))
    cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),2,cv2.LINE_AA)
    p0 = ClosestPointOnLine(np.array([x1,y1]),
                            np.array([x2,y2]),
                            np.array([cu,cv]))
    uvh = np.array([p0[0], p0[1], 1.0])
    proj = IK.dot(uvh)
    #print("proj:", proj, proj/np.linalg.norm(proj))
    dot_product = np.dot(np.array([0,0,1]), proj/np.linalg.norm(proj))
    pitch = np.arccos(dot_product) * r2d
    if p0[1] < cv:
        pitch = -pitch
    cv2.circle(frame,(int(p0[0]), int(p0[1])), 5, (255,0,255), 1, cv2.LINE_AA)
    cv2.circle(frame,(int(cu), int(cv)), 10, (255,0,255), 2, cv2.LINE_AA)
    cv2.line(frame, (int(p0[0]), int(p0[1])), (int(cu),int(cv)),(255,0,255), 1, cv2.LINE_AA)
