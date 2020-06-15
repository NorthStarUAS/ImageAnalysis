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

# locate horizon and estimate relative roll/pitch of the camera
def horizon(frame, IK, cu, cv):
    # attempt to threshold on high blue values (blue<->white)
    b, g, r = cv2.split(frame)
    cv2.imshow("b", b)
    #cv2.imshow("g", g)
    #cv2.imshow("r", r)
    #print("shape:", frame.shape)
    #print("blue range:", np.min(b), np.max(b))
    #print('ave:', np.average(b), np.average(g), np.average(r))

    # Otsu thresholding on blue channel
    ret2, thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print('ret2:', ret2)
    cv2.imshow('otsu mask', thresh)

    # dilate the mask a small bit
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
  
    # global thresholding on blue channel before edge detection
    #thresh = cv2.inRange(frame, (210, 0, 0), (255, 255, 255))
    #cv2.imshow('global mask', thresh)
        
    preview = cv2.bitwise_and(frame, frame, mask=thresh)
    cv2.imshow("threshold", preview)

    # the lower the 1st canny number the more total edges are accepted
    # the lower the 2nd canny number the less hard the edges need to
    # be to accept an edge
    edges = cv2.Canny(b, 50, 150)
    #edges = cv2.Canny(b, 200, 600)
    cv2.imshow("edges", edges)

    # Use the blue mask (Otsu) to filter out edge noise in area we
    # don't care about (to improve performance of the hough transform)
    edges = cv2.bitwise_and(edges, edges, mask=thresh)
    cv2.imshow("masked edges", edges)
    
    #theta_res = np.pi/180       # 1 degree
    theta_res = np.pi/1800      # 0.1 degree
    threshold = int(frame.shape[1] / 8)
    if True:
        # Standard hough transform.  Presumably returns lines sorted
        # by most dominant first
        lines = cv2.HoughLines(edges, 1, theta_res, threshold)
        for line in lines[:1]:  # just the 1st/most dominant
            #print(line[0])
            rho, theta = line[0]
            #print("theta:", theta * r2d)
            roll = 90 - theta*r2d
            # this will be wrong, but maybe approximate right a little bit
            if np.abs(theta) > 0.00001:
                m = -(np.cos(theta) / np.sin(theta))
                b = rho / np.sin(theta)
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
            print("roll: %.1f pitch: %.1f" % (roll, pitch))
            cv2.circle(frame,(int(p0[0]), int(p0[1])), 5, (255,0,255), 1, cv2.LINE_AA)
            cv2.circle(frame,(int(cu), int(cv)), 10, (255,0,255), 2, cv2.LINE_AA)
            cv2.line(frame, (int(p0[0]), int(p0[1])), (int(cu),int(cv)),(255,0,255), 1, cv2.LINE_AA)
            
    else:
        # probabalistic hough transform (faster?) but more trouble
        # seeing through gaps
        lines = cv2.HoughLinesP(edges, 1, theta_res, threshold, maxLineGap=50)
        if not lines is None:
            for line in lines[:1]:
                for x0,y0,x1,y1 in line:
                    #print("theta:", theta * r2d, "roll:", 90 - theta * r2d)
                    # this will be wrong, but maybe approximate right a little bit
                    #if np.abs(theta) > 0.00001:
                    #    m = -(np.cos(theta) / np.sin(theta))
                    #    b = rho / np.sin(theta)
                    #a = np.cos(theta)
                    #b = np.sin(theta)
                    #x0 = a*rho
                    #y0 = b*rho
                    #print("p0:", x0, y0)
                    #len2 = 1000
                    #x1 = int(x0 + len2*(-b))
                    #y1 = int(y0 + len2*(a))
                    #x2 = int(x0 - len2*(-b))
                    #y2 = int(y0 - len2*(a))
                    cv2.line(frame,(x0,y0),(x1,y1),(255,0,255),2,cv2.LINE_AA)

    cv2.imshow("horizon", frame)
    return roll, pitch
        
