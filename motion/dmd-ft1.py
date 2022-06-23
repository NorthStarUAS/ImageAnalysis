#!/usr/bin/env python3

# explore using dmd on a small set of inputs, but a high number of samples

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time

from pydmd import DMD

parser = argparse.ArgumentParser(description='dmd scale test.')
args = parser.parse_args()

# create some semi random data
#random.seed()
sensors = 6
samples = 1000

data = []
for i in range(samples):
    v = []
    for j in range(sensors):
        # try one
        if j < sensors - 1:
            val = math.sin(i / (8*j+1) + random.uniform(-1,1)) + random.uniform(j*0.5,j*0.55)
        else:
            val = v[0] * math.sin(v[1]) + v[2] * v[3] * v[4]
            print(val)
        # try two
        val = i*j + i
        v.append(val)
    data.append(v)

X = np.array(data[:-1]).T
Y = np.array(data[1:]).T
print("X:\n", np.array(X))
print("Y:\n", np.array(Y))

# Y = A * X, solve for A, now A is a matrix that projects v_n-1 -> v_n(est)

# X isn't nxn and doesn't have a direct inverse, so first perform an svd:
#
# Y = A * U * D * V.T

(u, s, vh) = np.linalg.svd(X, full_matrices=True)
print("u:\n", u.shape, u)
print("s:\n", s.shape, s)
print("vh:\n", vh.shape, vh)

print( "close?", np.allclose(X, np.dot(u * s, vh[:sensors, :])) )

# after algebraic manipulation
#
# A = Y * V * D.inv() * U.T

v = vh.T
print("s inv:", 1/s)

A = Y @ (v[:,:sensors] * (1/s)) * u.T
print("A:\n", A.shape, A)

plt.figure()
for j in range(sensors):
    plt.plot(X[j,:], label="%d" % j)
plt.legend()
plt.show()

pred = []
for v in data[:-1]:
    p = A @ v
    pred.append(p)

Ypred = np.array(pred).T

for j in range(sensors):
    plt.figure()
    plt.plot(Y[j,:], label="orig %d" % j)
    plt.plot(Ypred[j,:], label="pred %d" % j)
    plt.legend()
    plt.show()
 
# dmd options and structures
max_rank = int(samples * 0.1)
dmd = DMD(svd_rank=max_rank)
dmd.fit(np.array(X))

dmd.plot_eigs(show_axes=True, show_unit_circle=True)

idx = np.argsort(np.abs(dmd.eigs-1))
plt.figure()
for i in idx:
    dynamic = dmd.dynamics[i]
    #label = "%.4f%+.4fj" % (dmd.eigs[i].real, dmd.eigs[i].imag)
    label = "freq = %.4f" % (dmd.frequency[i])
    plt.plot(dynamic.real, label=label)
plt.legend()
plt.title('Dynamics')
plt.show()

def draw_mode(label, mode, shape, factor=2):
    real = factor * np.abs(mode.real)
    min = np.min(real)
    max = np.max(real)
    range = max - min
    equalized = (real - min) * (255 / range)
    (h, w) = shape[:2]
    big = cv2.resize(np.flipud(equalized.reshape((dmd_size,dmd_size)).astype('uint8')), (w, h), interpolation=cv2.INTER_AREA)
    draw_bar(big, min, max)
    cv2.imshow(label, big)
    return big

pbar = tqdm(total=int(total_frames), smoothing=0.05)
for frame in reader.nextFrame():
    counter += 1
    if counter < args.skip_frames:
        continue
    
    frame = frame[:,:,::-1]     # convert from RGB to BGR (to make opencv happy)
    #if counter % 2 != 0:
    #    continue
    
    frame_scale = cv2.resize(frame, (0,0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    cv2.imshow('scaled orig', frame_scale)
    frame_undist = cv2.undistort(frame_scale, K, np.array(dist))
    cv2.imshow("frame undist", frame_undist)

    # update the flow estimate
    M, prev_pts, curr_pts = flow.update(frame_undist)
    print("M:\n", M)
    
    #farneback.update(frame_undist)
    
    if M is None or prev_filt.shape[0] == 0 or curr_filt.shape[0] == 0:
        prev_filt = frame_undist.copy().astype('float32')
        curr_filt = frame_undist.copy().astype('float32')
        diff = cv2.absdiff(prev_filt, curr_filt)
        bg_filt = frame_undist.copy().astype('float32')
    else:
        prev_proj = frame_undist.copy()
        curr_proj = frame_undist.copy()
        bg_proj = frame_undist.copy()
        prev_proj = cv2.warpPerspective(prev_filt.astype('uint8'), M, (frame_undist.shape[1], frame_undist.shape[0]), prev_proj, flags=warp_flags, borderMode=cv2.BORDER_TRANSPARENT)
        curr_proj = cv2.warpPerspective(curr_filt.astype('uint8'), M, (frame_undist.shape[1], frame_undist.shape[0]), curr_proj, flags=warp_flags, borderMode=cv2.BORDER_TRANSPARENT)
        bg_proj = cv2.warpPerspective(bg_filt.astype('uint8'), M, (frame_undist.shape[1], frame_undist.shape[0]), bg_proj, flags=warp_flags, borderMode=cv2.BORDER_TRANSPARENT)
        curr_filt = curr_proj.astype('float32') * (1 - fg_alpha) \
            + frame_undist.astype('float32') * fg_alpha
        cv2.imshow("prev_filt", prev_filt.astype('uint8'))
        cv2.imshow("curr_filt", curr_filt.astype('uint8'))
        diff = cv2.absdiff(prev_proj.astype('uint8'), curr_filt.astype('uint8'))
        bg_filt = bg_proj.astype('float32') * (1 - bg_alpha) \
            + frame_undist.astype('float32') * bg_alpha
        prev_filt = curr_proj.astype('float32') * (1 - fg_alpha) \
            + frame_undist.astype('float32') * fg_alpha
    diff_max = np.max(diff)
    diff_factor = 0.95*diff_factor + 0.05*diff_max
    if diff_factor < diff_max:
        diff_factor = diff_max
    print("diff_factor:", diff_factor)
    diff_img = (255*diff.astype('float32')/diff_factor).astype('uint8')
    cv2.imshow("diff", diff_img.astype('uint8'))
    cv2.imshow("background", bg_filt.astype('uint8'))

    # now run dmd on the diff image (already compensated for camera
    # motion)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (dmd_size,dmd_size), interpolation=cv2.INTER_AREA)
    if not small.any():
        continue
    X.append( np.flipud(small) )
    while len(X) > window_size:
        del X[0]
    dmd.fit(np.array(X))
    if len(dmd.eigs):
        #print(dmd.eigs)
        idx = np.argsort(np.abs(dmd.eigs-1))
        #idx = np.argsort(np.abs(dmd.eigs.imag))
        print(idx)
        print(dmd.eigs)
        print(dmd.eigs[idx[0]])
        print(dmd.reconstructed_data.shape)

        big = 255 * dmd.reconstructed_data[:,-1] / np.max(dmd.reconstructed_data[:,-1]) # avoid overflow
        big = cv2.resize(np.flipud(big.reshape((dmd_size,dmd_size)).astype('uint8')), (frame_undist.shape[1], frame_undist.shape[0]), interpolation=cv2.INTER_AREA)
        big = 255 * ( big / np.max(big) )
        cv2.imshow("reconstructed", big.astype('uint8'))
        
        def draw_text_delete_me(img, label, x, y, subscale=1.0, just="center"):
            font_scale = subscale * h / 700
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 1)
            if just == "center":
                locx = int(x - size[0][0]*0.5)
                locy = int(y + size[0][1]*1.5)
            elif just == "lower-right":
                locx = int(x - size[0][0])
                locy = int(y - size[0][1])

            cv2.putText(img, label, (locx, locy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        1, cv2.LINE_AA)

        (h, w) = frame_undist.shape[:2]
        grid = np.zeros( (h*rows, w*cols) ).astype('uint8')
        orig_gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        grid[0:h,0:w] = orig_gray
        draw_text(grid, "Original", w*0.5, 0)
        r = 0
        c = 1
        for i in range(0, max_rank, 2):
            if i >= len(idx):
                break
            #print(i)
            if c >= cols:
                r += 1
                c = 0
            #print("grid:", r, c, "i:", i)
            if i == 0:
                factor = 1
            else:
                factor = 2
            grid[r*h:(r+1)*h,c*w:(c+1)*w] = draw_mode("a", dmd.modes[:,idx[i]], gray.shape, factor)
            #grid[r*h:(r+1)*h,c*w:(c+1)*w] = scaled
            eig = dmd.eigs[idx[i]]
            label = "Mode: %d (%.4f + %.4fj)" % (i, eig.real, eig.imag)
            draw_text(grid, label, (c+0.5)*w, r*h)
            c += 1
        draw_text(grid, "www.uav.aem.umn.edu", w*(rows-0.03), h*(cols-0.03), just="lower-right")
        cv2.imshow("grid", grid)
        if args.write_dmd:
            print("grid:", grid.shape)
            mode_writer.writeFrame(grid)

    if True:
        frame_feat = frame_undist.copy()
        for pt in curr_pts:
            cv2.circle(frame_feat, (int(pt[0][0]), int(pt[0][1])), 3, (0,255,0), 1, cv2.LINE_AA)
        for pt in prev_pts:
            cv2.circle(frame_feat, (int(pt[0][0]), int(pt[0][1])), 2, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('features', frame_feat)

    if args.write:
        # if rgb
        motion_writer.writeFrame(diff_img[:,:,::-1])
        bg_writer.writeFrame(bg_filt[:,:,::-1])
        feat_writer.writeFrame(frame_feat[:,:,::-1])
        # if gray
        #motion_writer.writeFrame(diff_img)
        #bg_writer.writeFrame(prev_filt)

    if args.write_quad:
        quad = np.zeros( (h*2, w*2, 3) ).astype('uint8')
        quad[0:h,0:w,:] = frame_undist
        quad[h:,0:w,:] = frame_feat
        quad[0:h,w:,:] = diff_img
        quad[h:,w:,:] = bg_filt
        draw_text(quad, "Original", w*0.5, 0)
        draw_text(quad, "Feature Flow", w*0.5, h)
        draw_text(quad, "Motion Layer", w*1.5, 0)
        draw_text(quad, "Background Layer", w*1.5, h)
        draw_text(quad, "www.uav.aem.umn.edu", 1.97*w, 1.97*h, subscale=0.5, just="lower-right")

        cv2.imshow("quad", quad)
        quad_writer.writeFrame(quad[:,:,::-1])

    if 0xFF & cv2.waitKey(1) == 27:
        break

    pbar.update(1)
pbar.close()

cv2.destroyAllWindows()

