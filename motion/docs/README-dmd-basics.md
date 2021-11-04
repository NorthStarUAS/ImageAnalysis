# DMD - Dynamic Mode Decomposition

DMD simultaneously computes a set of fourier series approximations for
a set of input sensor signals. The solution will share a common set of
harmonic frequencies, but each sensor will have a unique set of
weightings.

Notice that for each individual sensor, the fourier series
approximation will not be as accurate as computing a unique set of
frequency harmonics.  However, DMD offers the advantage of exposing
common dominant frequency information (modes) across the entire array
of sensors.  When we plot mode information in it's correct spatial
relationship, we can begin to visualize motion characteristics in the
data set.  This technique can be most excellent for visualizing fluid
flows.

##

Consider a system that has "n" sensors which are sampled at "m" time
steps and we solve for "k" harmonic frequencies.  DMD will compute a
set of "n" weightings for each of the "k" harmonic frequencies.  The
harmonic frequencies can be evaluated at each of the "m" time steps.

We can refer to a set of "n" weightings associated with a single
harmonic frequency as a "mode."  We can refer to the harmonic
frequency of each mode evaluated at each time step as the "dynamics."

## Reconstructing the original sensor data

Once the DMD modes and dynamics have been computed, these can be used
to reconstruct an approximation of the original data set at any time
step.

To reconstruct an approximation at each time step, simply sum the
product of each mode multiplied by it's respective harmonic frequency
function evaluated at that time step.  equation: x'(t) = sum (for all
the modes): mode(i) * dynamics(i)

Notice the reconstructed fit for each individual sensor with DMD will
not be as accurate as if a fourier series was computed for each sensor
independently due to the shared harmonic frequencies in the DMD
solution.

## The modes

The modes (the set of sensor weightings for each harmonic frequency)
themselves can provide insight into the motion of the system.  In
fluid dynamics the modes can show the structure of the motion of
complex flow systems.

## Video and DMD

If we replace 'sensor' with 'pixel', DMD can be used to compute an
approximation to video data and possibly gain insight into the
structure of the motion in the video.  As before, we can select the
number of modes for the DMD solution.  This is the number of terms in
the fourier series, so increasing this value leads to a more accurate
approximation but more storage requirements and longer compute times.
As before the DMD solution can be used to reconstruct the data set at
any time step by computing the sum of each mode multiplied by its
respective dynamics.

Taking a step back, what does DMD offer when processing video?

* DMD computes a fourier series for each pixel using a single set of
  harmonic frequencies that are shared between all pixels.

* The DMD solution can be used to recreate an approximation to the
  original video (using a much smaller amount of data.)

* Each mode (the set of weightings for each pixel associated with one
  frequency) may provide some insight into the structure of motion
  within the video.

## Impulse response

Let's pause a moment to think about how a fourier series approximates
an impulse (or step function.)

    The only time-domain signal that contains all single-frequency
    elements with unit magnitude is an impulse (delta function)
    (https://onscale.com/blog/impulse-response-modeling-in-onscale/)

[ draw on intuition of how it requires infinite terms to approximate a
square wave or step function using a fourier series. ]

## The (really) bad news.

The nature of video is that from the perspective of individual pixels:
the sequence of values each pixel assumes over time are best
characterized as a sequence of random step function changes at random
times.  This can be shown with some effort to by playing a video and
sampling the value of specific pixels over time and plotting it.  Some
key insight can be gained by doing this step.

## The fixed base camera hack

If a segment of video is capture from a camera that is fixed in space
(not translating, rotating, or zooming) then we can observe that the
DMD zero frequency mode corresponds to the non-changing background in
the video.

If you plot the value of any of these background pixels over time,
they remain realtively constant (but may vary slightly due to sensor
noise or change in lighting conditions or camera exposure.)

The zero frequency DMD mode is the background portion of the scene.
The scene can be segmented into background (static) and foreground
(moving) by simply subtracting the background from the current frame.
What remains is the moving part of the scene.

Note that because frequency information loses meaning when modeling
impulse changes with a fourier series, the key insight here is not
what specific frequencies are found in the data, but the separation of
the zero frequency mode versus all the other modes.

## Consider a moving camera

Now imagine a camera on a UAV that is going through some combination
of translation and rotation.

We now know:

* DMD is computing a fourier series approximation for each pixel using
  a shared set of harmonic frequencies, but individual (per pixel)
  weightings.

* Pixel value changes over time in video most closely approximate step
  or impulse changes and generally do not have meaningful frequency
  information.

* Now as the camera is moving, all the pixels in the video are subject
  to random impulse changes.

At first glance, there is a hope that some subset of frequencies would
correspond to the movement of the scene due to camera motion.  We can
look at video and observe pixels moving though the scene so
intuitively it seems like there must be some useful frequency domain
information we could extract from DMD.

However, pixels or groups of pixel are not actually moving through the
scene, this is just an animation illusion created by independently
changing the values of individual pixels.  Our brain connects the dots
and does the rest.

Thus we need to go back to considering the value of each individual
pixel over time as approximated by DMD using a shared set of harmonic
frequencies.  In this case, sharing harmonic frequencies means that
the fourier approximation is less accurate than if each individual
pixel was approximated independently.

Still, can we look at the modes (the per-pixel weightings for each
harmonic frequency) and gain insight into the motion of the scene?  I
argue the answer is no in the general case of drone style surveillance
video.  More specific cases certainly could exist where the modes may
be useful, but not in the domain of general purpose drone video.  The
reasons include: (1) pixel value changes over time most closely
resemble impulse changes or step functions (2) all the pixels in the
scene are subject to these unpredicable step changes when the camera
moves (there are no pixels that maintain a constant value through the
segment of video.)

