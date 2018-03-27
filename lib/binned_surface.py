import numpy as np
import scipy.stats

# build an approximate surface model from scattered x,y,z data.  Uses
# scipy binned_statistic_2d

class binned_surface:
    def __init__(self):
        pass

    def make(self, xfit, yfit, zfit, bins=32):
        if bins < 1:
            bins = 1
        self.bins = bins
        self.mean, self.xedges, self.yedges, binnumber = scipy.stats.binned_statistic_2d(xfit, yfit, zfit, bins=self.bins, statistic='mean')
        self.count, self.xedges, self.yedges, binnumber = scipy.stats.binned_statistic_2d(xfit, yfit, zfit, bins=self.bins, statistic='count')
        self.xmin = self.xedges[0]
        self.xmax = self.xedges[-1]
        self.xrange = self.xmax - self.xmin
        self.xstep = self.xrange / float(self.bins)
        self.ymin = self.yedges[0]
        self.ymax = self.yedges[-1]
        self.yrange = self.ymax - self.ymin
        self.ystep = self.yrange / float(self.bins)
        print('bins:', bins)

    # incrementally fill the empty bins based on a weighted average of
    # neighbors
    def fill(self):
        done = False
        while not done:
            done = True
            tmp_mean = np.copy(self.mean)
            tmp_count = np.copy(self.count)
            for i in range(self.bins):
                for j in range(self.bins):
                    if np.isnan(self.mean[i][j]):
                        # print("nan:", i, j)
                        nsum = 0
                        ncount = 0
                        ncells = 0
                        for ii in range(i-1,i+2):
                            for jj in range(j-1,j+2):
                                if ii < 0 or ii >= self.bins:
                                    continue
                                if jj < 0 or jj >= self.bins:
                                    continue
                                if not np.isnan(self.mean[ii][jj]):
                                    nsum += self.mean[ii][jj]*self.count[ii][jj]
                                    ncount += self.count[ii][jj]
                                    ncells += 1
                        if ncount > 0:
                            tmp_mean[i][j] = nsum / float(ncount)
                            tmp_count[i][j] = int(ncount / ncells)
                            # print('filled in:', i, j, self.mean[i][j])
                            done = False
            self.mean = tmp_mean
            self.count = tmp_count

    # query the aproximated surface elevation at the requested
    # location.  return None if out of bounds
    def query(self, x, y):
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return None
        c = int((x - self.xmin) / self.xstep)
        r = int((y - self.ymin) / self.ystep)
        if c < 0: c = 0
        if c >= self.bins - 1: c = self.bins - 1
        if r < 0: r = 0
        if r >= self.bins - 1: r = self.bins - 1
        return self.mean[c][r]
        
    def intersect(self, ned, v, avg_ground):
        p = ned[:] # copy hopefully

        # sanity check (always assume camera pose is above ground!)
        if v[2] <= 0.0:
            return p

        eps = 0.01
        count = 0
        print("start:", p)
        print("vec:", v)
        print("ned:", ned)
        surface = self.query(p[0], p[1])
        if surface == None:
            print(" initial surface interp returned none")
            surface = avg_ground
        error = abs(p[2] - surface)
        print("  p=%s surface=%s error=%s" % (p, surface, error))
        while error > eps and count < 25 and surface <= 0:
            d_proj = -(ned[2] - surface)
            factor = d_proj / v[2]
            n_proj = v[0] * factor
            e_proj = v[1] * factor
            print("proj = %s %s" % (n_proj, e_proj))
            p = [ ned[0] + n_proj, ned[1] + e_proj, ned[2] + d_proj ]
            print("new p:", p)
            surface = self.query(p[0], p[1])
            if surface == None:
                print("interpolation went out of bounds, not continuing")
                break
            error = abs(p[2] - surface)
            print("  p=%s surface=%.2f error = %.3f" % (p, surface, error))
            count += 1
        print("surface:", p)
        if p[2] > -10000 and p[2] < 0:
            return p
        else:
            print(" returning nans")
            return np.zeros(3)*np.nan

    def intersect_vectors(self, ned, v_list, avg_ground):
        pt_list = []
        for v in v_list:
            p = self.intersect(ned, v.flatten(), avg_ground)
            pt_list.append(p)
        return pt_list
