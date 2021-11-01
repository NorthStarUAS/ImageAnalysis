# Thoughts on using DMD from a moving camera

## Adjacency (in space)

I think it's important to clarify the role of physical adjacency of
sensors/pixels in DMD.  Our eyes and human brains can easily find
groups of similar pixels.  However, it is our brain that is doing this
work.  Individual pixels can assume any value without considering the
value of adjacent pixels.

For DMD, the order of the input vector data elements does not affect
the correct output of the algorithm.  If the input is an image, the
grid of pixels can be stacked by columns, stacked by rows, or even
assigned a "random" order, as long as this is done consistently for
each time step, and the arrangement is reversed before interpreting,
plotting, visualizing the results.

The DMD algorithm finds a best fit set of modes for the sensor/sample
behavior.  Sensors are grouped in the sense of "like behavior" but not
in the sense of physical location or adjacency.

## Visual motion of groups of pixels (aka blobs)

Another point that is important to clarify is also related to
adjacency.  Consider the apparent motion of a group of adjacent
pixels.  Our eyes/brain does scene segmentation and motion tracking
very well.  However, from the perspective of an animated video, each
frame of the video is simply an array of pixels that can take any
value independent of their neighbors.  Video motion is simply an
animation of individual frames of input being shown in quick
succession.

From the perspective of an individual pixel, it's value changes over
time, but the pixel itself is unaware of larger structures in each
frame or the frame-to-frame motion of the video.

If we plot the value of an individual pixel from the video animation
in the time domain, the plot will more closely match a step function
(or a sequence of step functions) than any other type of function.

The DMD algorithm is perfoming this transformation to the frequency
domain for all the pixels simultaneously and finding a set of modes
that best fits the collection of step functions.

## DMD-based scene segmentation in video from a static camera

When the camera is not moving, pixels representing a static background
portion of a scene should not change (or change very little due to
sensor noise, etc.)

Applying DMD to "static" video produces a zero frequency mode
corresponding to the non-changing pixels in the video.  We call this
the "background."  (Notice the DMD zero frequency mode maps directly
to simply averaging the frames of video together.)

The moving portion of the video can then be isolated by subtracting
the background from the current frame of video and whatever is left
over is considered the moving portion.

This works well, and DMD is a success in this use case.  Frequency
information correspondes to no motion (sum of near-zero frequency
modes) or some motion (sum of non-zero frequency modes).  But notice
that from the perspective of an individual pixel, we cannot extract
much useful information beyond zero frequency vs. non-zero frequency.
Due to the step function nature of individual pixel changes, the modes
do not convey useful information about the change, only that something
has changed.  (Note: for general purpose video, not for fluids
analysis.)

Also observe that simpy averaging frames is an O(n) operation and also
achieves the same result as computing the DMD zero frequency mode, so
much faster than DMD and scales up in a much friendlier way.

## DMD-based scene segmentation in video from a moving camera

Now consider that the camera is moving.  The static background will
appear to be moving in the video.  Our eyes/brain will do a good job
of interpreting this and understanding the scene.

However, DMD is mapping the values of each individual pixel to the
frequency domain, so we need to consider the camera motion from the
perspective of an individual pixel.  Again, similar to motion in a
static camera, motion of the background (for the perspective of an
individual pixel) acts more like a step function than any other
function.

Because the camera is moving and all the pixels are now subject to
change, the trick used in the fixed-camera use-case no longer works
and can't be directly extended in any useful way.  This is bad news,
but not the end of the story.
