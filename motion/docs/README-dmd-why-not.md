## Proof by [counter] example: DMD with Moving Drone Video

Let's just jump straight to a visual example (youtube video.)
Consider a short video clip of a moving vehicle captured while a drone
is also moving.  The original video is shown in the upper left, the
"zero-frequency" moe is shown upper center, and the next 7 modes
complete the grid.

DMD is computed with a 64 frame sliding window (roughly 2 seconds.)
The video dimensions are scaled down to a 200x200 array to reduce
memory usage in the DMD algorithm.  Notice the drone starts
motionless, but then begins to move a few seconds into the video.

If you watch this video all the way through, do any of these modes
show a separation of steady background versus moving vehicle?  Do any
of the modes show the background isolated from the foreground, or visa
versa?

##

Only using Fourier frequencies or weights (modes) computed for each
sensor position, could that information be used to separate out the
dynamics of the system versus the dynamics of the sensor?  This is the
primary question.  My goal of this paper is to demonstrate that this
is not possible in any general case.

Jumping ahead, because DMD is simply a Fourier series approximation to
the value of a set of sensors over time, what does that mean if
sensors can move arbitrarily (possibly unpredictably and unmeasurably)
through the system?  The signal at each sensor will be approximated,
but does that signal hold meaningful information in the frequency
domain when the sensor is moving through the system?

Notice that for each individual sensor, the DMD-based Fourier series
approximation will not be as accurate as computing the Fourier series
for each individual sensor due to the additional constraint of sharing
a common set of basis functions.  However, DMD offers the advantage
of exposing common dominant frequency information (modes) across the
entire array of sensors.  When we plot mode information in it's
correct spatial relationship, we can begin to visualize motion
characteristics in the data set.
# Consider a moving camera

Now imagine a camera on a UAV that is going through some combination
of translation and rotation.

We now know:

* DMD is computing a Fourier series approximation for each pixel using
  a shared set of frequencies, but with individual (per pixel)
  weightings.

* Pixel value changes over time in video most closely approximate step
  or impulse changes and generally do not have meaningful frequency
  information beyond changing vs. not changing.

* Translating or rotating the camera is similar to changing sensor
  positions in a fluids experiment.  The sensor is now sampling a
  different point in the system.

At first glance, there is a hope that some subset of DMD
frequencies/modes would correspond to the movement of the scene due to
camera motion.  We can look at video and observe pixels moving though
the scene so intuitively it seems like there must be some useful
frequency domain information we could extract from DMD.

However, an important distinction is that pixels or groups of pixel
are not actually moving through the scene, this is just an animation
illusion created by independently changing the values of individual
pixels.  Our eyes/brain connects the dots and does the rest.

DMD (Fourier series approximation) is approximating the time series
for each individual pixel.

Still, can we look at the modes (the per-pixel weightings for each
frequency) and gain insight into the motion of the scene?  I argue the
answer is no in the general case of drone style surveillance video.
(1) The weightings are convoluted with pixel brightness (2) The time
series change in any pixel value is not a periodic function.

We can still segment changing pixels from non changing pixels, but in
moving video, generally all the pixels are changing so this is not a
useful distinction.

# Moving Video Example

Consider a video with a dynamic moving camera (chase quad) and a
dynamic subject (aerobatic aircraft.)

Visually you can see the original video frame in the first grid
location.  Then moving to the right you see the zero frequency mode
(top center) which visually looks very much like an average of the
input frames (as we would expect).  But because the camera is moving,
the input frames get smeared together (as we would expect.)

Looking at the other non-zero frequency modes, we can see the all the
elements described above.  All the motion shows up in all the modes,
the edges of areas show up as step changes as they "move" through the
scene.

In the case of a stationary camera we can look at the zero frequency
mode and observe it is the background without the moving elements.

In the case of a moving camera we can view all the different modes
(and animate them throughout the course of the moving video), but we
do not see any modes that contain clear information with respect to
segmenting the moving parts of the scene from the static background.
The information we were hoping to extract just isn't there in the way
we hoped.  Hopefully the information and background provided
throughout this document kept this from being a surprise.

![motion modes](./motion-modes.png)

# Conclusion and future work

If the primary goal is to segment a challenging and highly dynamic
scene such as shown in the final example, then this isn't the end of
the story.  But I hope I have shown that DMD may not provide
sufficient information to be a large part of the answer.