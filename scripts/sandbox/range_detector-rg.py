#!/usr/bin/python3
# -*- coding: utf-8 -*-

# USAGE: You need to specify a filter and "only one" image source
#
# (python) range-detector --filter RGB --image /path/to/image.png
# or
# (python) range-detector --filter HSV --webcam

import cv2
import numpy as np
import argparse
from operator import xor
import skvideo.io               # pip3 install sk-video


def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range filter. RGB or HSV')
    ap.add_argument('-i', '--image', required=False,
                    help='Path to the image')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('--video', help='Specify a video as input')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image after applying the mask',
                    action='store_true')
    args = vars(ap.parse_args())

    #if not xor(bool(args['image']), bool(args['webcam'])):
    #    ap.error("Please specify only one image source")

    if not args['filter'].upper() in ['RGB', 'HSV', 'GGG']:
        ap.error("Please speciy a correct filter.")

    return args


def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values


def main():
    args = get_arguments()

    range_filter = args['filter'].upper()

    if args['image']:
        image = cv2.imread(args['image'])
        image = cv2.GaussianBlur(image, (5,5), 4)
        
        if range_filter == 'RGB':
            frame_to_thresh = image.copy()
        elif range_filter == 'HSV':
            frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(frame_to_thresh)
            hue = np.mod((hue.astype('float') + 90), 180).astype('uint8')
            frame_to_thresh = cv2.merge( (hue, sat, val) )
        else:
            g, b, r = cv2.split(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            g[g==0] = 1                 # protect against divide by zero
            ratio = (r / g).astype('float') * 0.25
            # knock out the low end
            lum = gray.astype('float') / 255
            lumf = lum / 0.15
            lumf[lumf>1] = 1
            ratio *= lumf
            ratio[ratio>1] = 1
            index = (ratio*255).astype('uint8')
            frame_to_thresh = cv2.merge( (index, index, index) )

    elif args['video']:
        print("Opening ", args['video'])
        reader = skvideo.io.FFmpegReader(args['video'], inputdict={}, outputdict={})
    else:
        camera = cv2.VideoCapture(0)

    setup_trackbars(range_filter)

    while True:
        if args['video']:
            for image in reader.nextFrame():
                image = image[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)

                #frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                frame_to_thresh = image.copy()
                h, s, v = cv2.split(frame_to_thresh)
                cv2.imshow('value', v)
                print(np.average(v))
                v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
                thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
                preview = cv2.bitwise_and(image, image, mask=thresh)
                cv2.imshow("Preview", preview)
                if cv2.waitKey(1) & 0xFF is ord('q'):
                    break
        elif args['webcam']:
            ret, image = camera.read()

            if not ret:
                break

            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        if args['preview']:
            preview = cv2.bitwise_and(image, image, mask=thresh)
            cv2.imshow("Preview", preview)
        else:
            cv2.imshow("Original", image)
            cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break


if __name__ == '__main__':
    main()
