from panda3d.core import LineSegs, NodePath

# draw a reticle on screen as a 3d object that is kept in line with camera center

class Reticle():
    def __init__(self, render):
        self.render = render
        self.last_cam_pos = [0, 0, 0]

    def build(self, cam_pos, view_size):
        alpha = 0.5
        depth = 0
        a1 = view_size / 20
        a2 = view_size / 5
        
        ls = LineSegs()
        ls.setThickness(1)
        ls.setColor(0.0, 1.0, 0.0, alpha)
        ls.moveTo(cam_pos[0] + a1, cam_pos[1], 0)
        ls.drawTo(cam_pos[0] + a2, cam_pos[1], 0)
        ls.moveTo(cam_pos[0] - a1, cam_pos[1], 0)
        ls.drawTo(cam_pos[0] - a2, cam_pos[1], 0)
        ls.moveTo(cam_pos[0], cam_pos[1] + a1, 0)
        ls.drawTo(cam_pos[0], cam_pos[1] + a2, 0)
        ls.moveTo(cam_pos[0], cam_pos[1] - a1, 0)
        ls.drawTo(cam_pos[0], cam_pos[1] - a2, 0)
        self.node = NodePath(ls.create())
        self.node.setDepthTest(False)
        self.node.setDepthWrite(False)
        self.node.setBin("unsorted", depth)
        self.node.reparentTo(self.render)

    def delete(self):
        if hasattr(self, 'node'):
            self.node.removeNode()
        
    def update(self, cam_pos, view_size):
        self.delete()
        self.build(cam_pos, view_size)
        
