#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv3/lib/python2.7/site-packages/")
import cv2

import argparse
import fnmatch
import numpy as np
from matplotlib import pyplot as plt
import os

parser = argparse.ArgumentParser(description='Image experiments.')
parser.add_argument('--image', help='image name')
parser.add_argument('--dir', help='image directory')
parser.add_argument('--scale', type=float, help='scale')
parser.add_argument('--method', default='equalize',
                    choices=['equalize', 'boost-red1', 'boost-red2',
                             'red-blue'])
args = parser.parse_args()

def my_scale(image, scale):
    result = cv2.resize(image, (0,0), fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
    return result

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
def my_equalize(channel, method='clahe'):
    #print channel.min(), channel.max()
    if method == 'simple':
        result = cv2.equalizeHist(channel)
    elif method == 'clahe':
        result = clahe.apply(channel)
    #print result.min(), result.max()
    return result

def analyze_image(name):
    img = cv2.imread(name)

    if args.scale:
        img = my_scale(img, args.scale)

    cv2.imshow('original', img)
    
    if args.method == 'equalize':
        # BGR equalization
        b, g, r = cv2.split(img) 
        clb = my_equalize(b)
        clg = my_equalize(g)
        clr = my_equalize(r)
        #cv2.imshow('equalize blue', clb)
        #cv2.imshow('equalize green', clg)
        cv2.imshow('equalize red', clr)
        result = cv2.merge((clb,clg,clr))
        
    elif args.method == 'boost-red1':
        # switch to HSV color space where red: hue = 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        print 'h range:', h.min(), h.max()
        print 's range:', s.min(), s.max()

        clh = my_equalize(h)
        cls = my_equalize(s)
        clv = my_equalize(v)
        equ_hsv = cv2.merge((h,cls,clv))
        result = cv2.cvtColor(equ_hsv, cv2.COLOR_HSV2BGR)

    elif args.method == 'boost-red2':
        # switch to HSV color space where red: hue = 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        print 'h range:', h.min(), h.max()
        print 's range:', s.min(), s.max()

        # simple thresholding
        hinv = 255 - h
        ret,thresh_h = cv2.threshold(hinv, thresh=224, maxval=255,
                                     type=cv2.THRESH_TOZERO)
        cv2.imshow('h', h)
        cv2.imshow('thresh_h', thresh_h)
        
        ret,thresh_s = cv2.threshold(s, thresh=64, maxval=255,
                                     type=cv2.THRESH_TOZERO)
        cv2.imshow('s', s)
        cv2.imshow('thresh_s', thresh_s)
    
        #clh = my_equalize(h)
        #cv2.imshow('h equ', clh)

        # clh = my_equalize(thresh_h)
        # max = clh.max()
        # clh = cv2.convertScaleAbs(clh * (255.0 / max))
        # cls = my_equalize(thresh_s)
        # my_equalize(cls)
        # max = cls.max()
        # cls = cv2.convertScaleAbs(cls * (255.0 / max))
        # clv = my_equalize(v)
        # my_equalize(clv)
        # max = clv.max()
        # clv = cv2.convertScaleAbs(clv * (255.0 / max))
        #cv2.imwrite('clahe-h.jpg', clh)
        #cv2.imwrite('clahe-s.jpg', cls)
        #cv2.imwrite('clahe-v.jpg', clv)
        #equ_hsv = cv2.merge((clh,cls,clv))
        #equ_bgr = cv2.cvtColor(equ_hsv, cv2.COLOR_HSV2BGR)
        #cv2.imwrite('equ_hsv.jpg', equ_bgr)

        mask = cv2.multiply(hinv, thresh_s, None, 1.0/255.0)
        print 'mask range:', mask.min(), mask.max()
        max = mask.max()
        mask = cv2.convertScaleAbs(mask * (255.0 / max))
        cv2.imshow('mask', mask)
        #cv2.imwrite('mask.jpg', mask)

        reshinv = cv2.multiply(hinv, mask, None, 1.0/255.0)
        max = reshinv.max()
        reshinv = cv2.convertScaleAbs(reshinv * (255.0 / max))
        cv2.imshow('reshinv', reshinv)

        ress = cv2.multiply(s, mask, None, 1.0/255.0)
        max = ress.max()
        ress = cv2.convertScaleAbs(ress * (255.0 / max))
        
        resv = cv2.multiply(v, mask, None, 1.0/255.0)
        max = resv.max()
        resv = cv2.convertScaleAbs(resv * (255.0 / max))

        equ_hsv = cv2.merge((h,ress,resv))
        result = cv2.cvtColor(equ_hsv, cv2.COLOR_HSV2BGR)
        
    if args.method == 'red-blue':
        # BGR equalization
        b, g, r = cv2.split(img) 
        #cv2.imshow('blue', b)
        #cv2.imshow('green', g)
        #cv2.imshow('red', r)
        rb = np.divide(r.astype('float'), b.astype('float'))
        rg = np.divide(r.astype('float'), g.astype('float'))
        #r = rb + rg
        #print ' min:', np.min(np.ravel(r))
        #print ' max:', np.max(np.ravel(r))
        nrb = cv2.normalize(rb.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) * 255;
        nrb = cv2.convertScaleAbs(nrb)
        nrg = cv2.normalize(rg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) * 255;
        nrg = cv2.convertScaleAbs(nrg)
        #cv2.imshow('red-blue', nrb)
        #cv2.imshow('red-green', nrg)

        cv2.imwrite('rgb.jpg', img)
        cv2.imwrite('rg.jpg', nrg)
        
        result = nrg

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

if args.image:
    result = analyze_image(args.image)
    cv2.imwrite('enhanced.jpg', result)
elif args.dir:
    files = []
    for file in os.listdir(args.dir):
        if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
            files.append(file)
            files.sort()
    for image in files:
        print image
        analyze_image(os.path.join(args.dir, image))
        
else:
    print "no input source specified"
