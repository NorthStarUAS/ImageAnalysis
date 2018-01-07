import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import cv2
import math

sys.path.append('../lib')
import ProjectMgr

# Draw a match entry.  Creates a window for each image referenced by
# the match.  A cropped portion of the image is drawn with the match
# location highlited.  Waits for a key press to continue, the key
# value is returned to the calling layer.
def draw_match(i, index, matches, image_list):
    green = (0, 255, 0)
    red = (0, 0, 255)

    match = matches[i]
    print('match:', match, 'index:', index)
    for j, m in enumerate(match[1:]):
        print(' ', m, image_list[m[0]])
        img = image_list[m[0]]
        kp = img.kp_list[m[1]].pt # distorted
        #kp = img.uv_list[m[1]]  # undistored
        print(' ', kp)
        rgb = img.load_rgb()
        h, w = rgb.shape[:2]
        crop = True
        range = 300
        if ( j == index ) or len(match[1:]) == 2:
            color = red
        else:
            color = green
        if crop:
            cx = int(round(kp[0]))
            cy = int(round(kp[1]))
            if cx < range:
                xshift = range - cx
                cx = range
            elif cx > (w - range):
                xshift = (w - range) - cx
                cx = w - range
            else:
                xshift = 0
            if cy < range:
                yshift = range - cy
                cy = range
            elif cy > (h - range):
                yshift = (h - range) - cy
                cy = h - range
            else:
                yshift = 0
            print('size:', w, h, 'shift:', xshift, yshift)
            rgb1 = rgb[cy-range:cy+range, cx-range:cx+range]
            cv2.circle(rgb1, (range-xshift,range-yshift), 2, color, thickness=2)
        else:
            scale = 790.0/float(w)
            rgb1 = cv2.resize(rgb, (0,0), fx=scale, fy=scale)
            cv2.circle(rgb1,
                       (int(round(kp[0]*scale)), int(round(kp[1]*scale))),
                       2, color, thickness=2)
        cv2.imshow(img.name + ' (%d)' % m[0], rgb1)
    print('waiting for keyboard input...')
    key = cv2.waitKey() & 0xff
    cv2.destroyAllWindows()
    return key

def show_outliers(result_list, matches, image_list):
    print("Show outliers...")
    mark_sum = 0
    sum = 0.0
    count = len(result_list)

    # numerically it is better to sum up a list of floating point
    # numbers from smallest to biggest (result_list is sorted from
    # biggest to smallest)
    for line in reversed(result_list):
        sum += line[0]
        
    # stats on error values
    print(" computing stats...")
    mre = sum / count
    stddev_sum = 0.0
    for line in result_list:
        error = line[0]
        stddev_sum += (mre-error)*(mre-error)
    stddev = math.sqrt(stddev_sum / count)
    print("mre = %.4f stddev = %.4f" % (mre, stddev))

    mark_list = []
    for line in result_list:
        # print "line:", line
        print("  outlier index %d-%d err=%.2f" % (line[1], line[2],
                                                  line[0]))
        result = draw_match(line[1], line[2], matches, image_list)
        if result == ord('d'):
            # add to delete feature
            mark_list.append( [line[1], line[2]] )
            mark_sum += 1
        elif result == 27 or result == ord('q'):
            # quit reviewing and go on to delete the marks
            break
    return mark_list

# mark the outlier
def mark_outlier(matches, match_index, feat_index, error):
    print('  outlier - match index:', match_index, 'feature index:', feat_index, 'error:', error)
    match = matches[match_index]
    match[feat_index+1] = [-1, -1]

def mark_using_list(mark_list, matches):
    for mark in mark_list:
        mark_outlier( matches, mark[0], mark[1], None )
        
# delete marked matches
def delete_marked_matches(matches):
    print(" deleting marked items...")
    for i in reversed(range(len(matches))):
        match = matches[i]
        has_bad_elem = False
        for j in reversed(range(1, len(match))):
            p = match[j]
            if p == [-1, -1]:
                has_bad_elem = True
                match.pop(j)
        if False and has_bad_elem: # was 'if args.strong and ...'
            print("deleting entire match that contains a bad element", i)
            matches.pop(i)
        elif len(match) < 3:
            print("deleting match that is now in less than 2 images:", match, i)
            matches.pop(i)
        elif False and len(match) < 4:
            # this is seeming like less and less of a good idea (Jan 3, 2017)
            print("deleting match that is now in less than 3 images:", match, i)
            matches.pop(i)
    print("final matches size:", len(matches))

