import pyglet
from pyglet.gl import *
 
win = pyglet.window.Window()
 
@win.event
def on_draw():
 
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT)
 
        # Draw some stuff
        glBegin(GL_POINTS)
        glVertex2i(50, 50)
        glVertex2i(75, 100)
        glVertex2i(100, 150)
        glEnd()
 
pyglet.app.run()

