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


