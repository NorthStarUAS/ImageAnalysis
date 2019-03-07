import numpy as np
import os
import pickle
import scipy.spatial

class Surface():
    def __init__(self, project_dir):
        surface_file = os.path.join(project_dir, 'models', 'surface.bin')
        if os.path.exists(surface_file):
            print("Loading surface:", surface_file)
            raw = pickle.load(open(surface_file, "rb"))
            print('Generating Delaunay mesh and interpolator ...')
            global_tri_list = scipy.spatial.Delaunay(np.array(raw['points']))
            self.interp = scipy.interpolate.LinearNDInterpolator(global_tri_list, raw['values'])
        else:
            self.interp = None

    def get_elevation(self, n, e):
        if not self.interp:
            return 0.0
        tmp = self.interp(n, e)
        if np.isnan(tmp):
            return 0.0
        else:
            return float(tmp)

            
