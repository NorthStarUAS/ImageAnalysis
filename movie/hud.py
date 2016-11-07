import datetime
import ephem
import math
import navpy
import numpy as np

# find our custom built opencv first
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")
import cv2

sys.path.append('../lib')
import transformations

# helpful constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi

class HUD:
    def __init__(self, K):
        self.K = K
        self.PROJ = None
        self.line_width = 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 0.6
        self.render_w = 0
        self.render_h = 0
        self.ref = [0.0, 0.0, 0.0]
        self.vel_filt = [0.0, 0.0, 0.0]

    def set_render_size(self, w, h):
        self.render_w = w
        self.render_h = h
        
    def set_line_width(self, line_width):
        self.line_width = line_width
        if self.line_width < 1:
            self.line_width = 1

    def set_font_size(self, font_size):
        self.font_size = font_size
        if self.font_size < 0.4:
            self.font_size = 0.4

    def set_ned_ref(self, lat, lon):
        self.ref = [ lat, lon, 0.0]

    def update_proj(self, PROJ):
        self.PROJ = PROJ
        
    def compute_sun_moon_ned(self, lon_deg, lat_deg, alt_m, timestamp):
        d = datetime.datetime.utcfromtimestamp(timestamp)
        #d = datetime.datetime.utcnow()
        ed = ephem.Date(d)
        #print 'ephem time utc:', ed
        #print 'localtime:', ephem.localtime(ed)

        ownship = ephem.Observer()
        ownship.lon = '%.8f' % lon_deg
        ownship.lat = '%.8f' % lat_deg
        ownship.elevation = alt_m
        ownship.date = ed

        sun = ephem.Sun(ownship)
        moon = ephem.Moon(ownship)

        sun_ned = [ math.cos(sun.az), math.sin(sun.az), -math.sin(sun.alt) ]
        moon_ned = [ math.cos(moon.az), math.sin(moon.az), -math.sin(moon.alt) ]

        return sun_ned, moon_ned

    def project_point(self, ned):
        uvh = self.K.dot( self.PROJ.dot( [ned[0], ned[1], ned[2], 1.0] ).T )
        if uvh[2] > 0.1:
            uvh /= uvh[2]
            uv = ( int(np.squeeze(uvh[0,0])), int(np.squeeze(uvh[1,0])) )
            return uv
        else:
            return None

    def draw_horizon(self, ned, frame):
        divs = 10
        pts = []
        for i in range(divs + 1):
            a = (float(i) * 360/float(divs)) * d2r
            n = math.cos(a)
            e = math.sin(a)
            d = 0.0
            pts.append( [n, e, d] )

        for i in range(divs):
            p1 = pts[i]
            p2 = pts[i+1]
            uv1 = self.project_point( [ned[0] + p1[0], ned[1] + p1[1], ned[2] + p1[2]] )
            uv2 = self.project_point( [ned[0] + p2[0], ned[1] + p2[1], ned[2] + p2[2]] )
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)

    def ladder_helper(self, q0, a0, a1, ned):
        q1 = transformations.quaternion_from_euler(-a1*d2r, -a0*d2r, 0.0, 'rzyx')
        q = transformations.quaternion_multiply(q1, q0)
        v = transformations.quaternion_transform(q, [1.0, 0.0, 0.0])
        uv = self.project_point( [ned[0] + v[0], ned[1] + v[1], ned[2] + v[2]] )
        return uv

    def draw_pitch_ladder(self, ned, frame, yaw_rad, beta_rad):
        a1 = 2.0
        a2 = 8.0
        #slide_rad = yaw_rad - beta_rad
        slide_rad = yaw_rad
        q0 = transformations.quaternion_about_axis(slide_rad, [0.0, 0.0, -1.0])
        for a0 in range(5,35,5):
            # above horizon

            # right horizontal
            uv1 = self.ladder_helper(q0, a0, a1, ned)
            uv2 = self.ladder_helper(q0, a0, a2, ned)
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label(frame, "%d" % a0, uv, self.font_size, self.line_width)
            # right tick
            uv1 = self.ladder_helper(q0, a0-0.5, a1, ned)
            uv2 = self.ladder_helper(q0, a0, a1, ned)
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)
            # left horizontal
            uv1 = self.ladder_helper(q0, a0, -a1, ned)
            uv2 = self.ladder_helper(q0, a0, -a2, ned)
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label(frame, "%d" % a0, uv, self.font_size, self.line_width)
            # left tick
            uv1 = self.ladder_helper(q0, a0-0.5, -a1, ned)
            uv2 = self.ladder_helper(q0, a0, -a1, ned)
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)

            # below horizon

            # right horizontal
            uv1 = self.ladder_helper(q0, -a0, a1, ned)
            uv2 = self.ladder_helper(q0, -a0-0.5, a2, ned)
            if uv1 != None and uv2 != None:
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                for i in range(0,3):
                    tmp1 = ( uv1[0] + int(0.375*i*du), uv1[1] + int(0.375*i*dv) )
                    tmp2 = ( tmp1[0] + int(0.25*du), tmp1[1] + int(0.25*dv) )
                    cv2.line(frame, tmp1, tmp2, (0,240,0), 1, cv2.CV_AA)
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label(frame, "%d" % a0, uv, self.font_size, self.line_width)

            # right tick
            uv1 = self.ladder_helper(q0, -a0+0.5, a1, ned)
            uv2 = self.ladder_helper(q0, -a0, a1, ned)
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)
            # left horizontal
            uv1 = self.ladder_helper(q0, -a0, -a1, ned)
            uv2 = self.ladder_helper(q0, -a0-0.5, -a2, ned)
            if uv1 != None and uv2 != None:
                du = uv2[0] - uv1[0]
                dv = uv2[1] - uv1[1]
                for i in range(0,3):
                    tmp1 = ( uv1[0] + int(0.375*i*du), uv1[1] + int(0.375*i*dv) )
                    tmp2 = ( tmp1[0] + int(0.25*du), tmp1[1] + int(0.25*dv) )
                    cv2.line(frame, tmp1, tmp2, (0,240,0), self.line_width, cv2.CV_AA)
                uv = ( uv1[0] + int(1.25*du), uv1[1] + int(1.25*dv) )
                self.draw_label(frame, "%d" % a0, uv, self.font_size, self.line_width)
            # left tick
            uv1 = self.ladder_helper(q0, -a0+0.5, -a1, ned)
            uv2 = self.ladder_helper(q0, -a0, -a1, ned)
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)

    def draw_flight_path_marker(self, ned, frame, pitch_rad, alpha_rad,
                                yaw_rad, beta_rad):
        q0 = transformations.quaternion_about_axis(yaw_rad + beta_rad, [0.0, 0.0, -1.0])
        a0 = (pitch_rad - alpha_rad) * r2d
        uv = self.ladder_helper(q0, a0, 0, ned)
        if uv != None:
            r1 = int(round(self.render_h / 60))
            r2 = int(round(self.render_h / 30))
            uv1 = (uv[0]+r1, uv[1])
            uv2 = (uv[0]+r2, uv[1])
            uv3 = (uv[0]-r1, uv[1])
            uv4 = (uv[0]-r2, uv[1])
            uv5 = (uv[0], uv[1]-r1)
            uv6 = (uv[0], uv[1]-r2)
            cv2.circle(frame, uv, r1, (0,240,0), self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv2, (0,240,0), self.line_width, cv2.CV_AA)
            cv2.line(frame, uv3, uv4, (0,240,0), self.line_width, cv2.CV_AA)
            cv2.line(frame, uv5, uv6, (0,240,0), self.line_width, cv2.CV_AA)

    def rotate_pt(self, p, center, a):
        x = math.cos(a) * (p[0]-center[0]) - math.sin(a) * (p[1]-center[1]) + center[0]

        y = math.sin(a) * (p[0]-center[0]) + math.cos(a) * (p[1]-center[1]) + center[1]
        return (int(x), int(y))

    def draw_vbars(self, ned, frame, yaw_rad, pitch_rad, ap_roll, ap_pitch):
        color = (186, 85, 211)      # medium orchid
        size = self.line_width
        a1 = 10.0
        a2 = 1.5
        a3 = 3.0
        q0 = transformations.quaternion_about_axis(yaw_rad, [0.0, 0.0, -1.0])
        a0 = ap_pitch

        # rotation point (about nose)
        rot = self.ladder_helper(q0, pitch_rad*r2d, 0.0, ned)

        # center point
        tmp1 = self.ladder_helper(q0, a0, 0.0, ned)
        center = rotate_pt(tmp1, rot, ap_roll*d2r)

        # right vbar
        tmp1 = self.ladder_helper(q0, a0-a3, a1, ned)
        tmp2 = self.ladder_helper(q0, a0-a3, a1+a3, ned)
        tmp3 = self.ladder_helper(q0, a0-a2, a1+a3, ned)
        uv1 = rotate_pt(tmp1, rot, ap_roll*d2r)
        uv2 = rotate_pt(tmp2, rot, ap_roll*d2r)
        uv3 = rotate_pt(tmp3, rot, ap_roll*d2r)
        if uv1 != None and uv2 != None and uv3 != None:
            cv2.line(frame, center, uv1, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, center, uv3, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv3, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv2, uv3, color, self.line_width, cv2.CV_AA)
        # left vbar
        tmp1 = self.ladder_helper(q0, a0-a3, -a1, ned)
        tmp2 = self.ladder_helper(q0, a0-a3, -a1-a3, ned)
        tmp3 = self.ladder_helper(q0, a0-a2, -a1-a3, ned)
        uv1 = rotate_pt(tmp1, rot, ap_roll*d2r)
        uv2 = rotate_pt(tmp2, rot, ap_roll*d2r)
        uv3 = rotate_pt(tmp3, rot, ap_roll*d2r)
        if uv1 != None and uv2 != None and uv3 != None:
            cv2.line(frame, center, uv1, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, center, uv3, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv3, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv2, uv3, color, self.line_width, cv2.CV_AA)

    def draw_heading_bug(self, ned, frame, ap_hdg):
        color = (186, 85, 211)      # medium orchid
        size = 2
        a = math.atan2(ve, vn)
        q0 = transformations.quaternion_about_axis(ap_hdg*d2r, [0.0, 0.0, -1.0])
        center = self.ladder_helper(q0, 0, 0, ned)
        pts = []
        pts.append( self.ladder_helper(q0, 0, 2.0, ned) )
        pts.append( self.ladder_helper(q0, 0.0, -2.0, ned) )
        pts.append( self.ladder_helper(q0, 1.5, -2.0, ned) )
        pts.append( self.ladder_helper(q0, 1.5, -1.0, ned) )
        pts.append( center )
        pts.append( self.ladder_helper(q0, 1.5, 1.0, ned) )
        pts.append( self.ladder_helper(q0, 1.5, 2.0, ned) )
        for i, p in enumerate(pts):
            if p == None or center == None:
                return
            #else:
            #    pts[i] = rotate_pt(pts[i], center, -cam_roll*d2r)
        cv2.line(frame, pts[0], pts[1], color, self.line_width, cv2.CV_AA)
        cv2.line(frame, pts[1], pts[2], color, self.line_width, cv2.CV_AA)
        cv2.line(frame, pts[2], pts[3], color, self.line_width, cv2.CV_AA)
        cv2.line(frame, pts[3], pts[4], color, self.line_width, cv2.CV_AA)
        cv2.line(frame, pts[4], pts[5], color, self.line_width, cv2.CV_AA)
        cv2.line(frame, pts[5], pts[6], color, self.line_width, cv2.CV_AA)
        cv2.line(frame, pts[6], pts[0], color, self.line_width, cv2.CV_AA)
        #pts = np.array( pts, np.int32 )
        #pts = pts.reshape((-1,1,2))
        #cv2.polylines(frame, pts, True, color, self.line_width, cv2.CV_AA)

    def draw_bird(self, ned, frame, yaw_rad, pitch_rad, roll_rad):
        color = (50, 255, 255)     # yellow
        size = 2
        a1 = 10.0
        a2 = 3.0
        a2 = 3.0
        q0 = transformations.quaternion_about_axis(yaw_rad, [0.0, 0.0, -1.0])
        a0 = pitch_rad*r2d

        # center point
        center = self.ladder_helper(q0, pitch_rad*r2d, 0.0, ned)

        # right vbar
        tmp1 = self.ladder_helper(q0, a0-a2, a1, ned)
        tmp2 = self.ladder_helper(q0, a0-a2, a1-a2, ned)
        uv1 = rotate_pt(tmp1, center, roll_rad)
        uv2 = rotate_pt(tmp2, center, roll_rad)
        if uv1 != None and uv2 != None:
            cv2.line(frame, center, uv1, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, center, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        # left vbar
        tmp1 = self.ladder_helper(q0, a0-a2, -a1, ned)
        tmp2 = self.ladder_helper(q0, a0-a2, -a1+a2, ned)
        uv1 = rotate_pt(tmp1, center, roll_rad)
        uv2 = rotate_pt(tmp2, center, roll_rad)
        if uv1 != None and uv2 != None:
            cv2.line(frame, center, uv1, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, center, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)

    filter_vn = 0.0
    filter_ve = 0.0
    tf_vel = 0.5
    def draw_course(self, ned, frame, vn, ve):
        global filter_vn
        global filter_ve
        color = (50, 255, 255)     # yellow
        size = 2
        filter_vn = (1.0 - tf_vel) * filter_vn + tf_vel * vn
        filter_ve = (1.0 - tf_vel) * filter_ve + tf_vel * ve
        a = math.atan2(filter_ve, filter_vn)
        q0 = transformations.quaternion_about_axis(a, [0.0, 0.0, -1.0])
        tmp1 = self.ladder_helper(q0, 0, 0, ned)
        tmp2 = self.ladder_helper(q0, 1.5, 1.0, ned)
        tmp3 = self.ladder_helper(q0, 1.5, -1.0, ned)
        if tmp1 != None and tmp2 != None and tmp3 != None :
            uv2 = rotate_pt(tmp2, tmp1, -cam_roll*d2r)
            uv3 = rotate_pt(tmp3, tmp1, -cam_roll*d2r)
            cv2.line(frame, tmp1, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, tmp1, uv3, color, self.line_width, cv2.CV_AA)

    def draw_label(self, frame, label, uv, font_scale, thickness, horiz='center', vert='center'):
            size = cv2.getTextSize(label, self.font, font_scale, thickness)
            if horiz == 'center':
                u = uv[0] - (size[0][0] / 2)
            else:
                u = uv[0]
            if vert == 'above':
                v = uv[1]
            elif vert == 'below':
                v = uv[1] + size[0][1]
            elif vert == 'center':
                v = uv[1] + (size[0][1] / 2)
            uv = (u, v)
            cv2.putText(frame, label, uv, self.font, font_scale, (0,255,0),
                        thickness, cv2.CV_AA)

    def draw_labeled_point(self, frame, ned, label, scale=1, vert='above'):
        uv = self.project_point([ned[0], ned[1], ned[2]])
        if uv != None:
            cv2.circle(frame, uv, 5, (0,240,0), 1, cv2.CV_AA)
        if vert == 'above':
            uv = self.project_point([ned[0], ned[1], ned[2] - 0.02])
        else:
            uv = self.project_point([ned[0], ned[1], ned[2] + 0.02])
        if uv != None:
            self.draw_label(frame, label, uv, scale, 1, vert=vert)

    def draw_lla_point(self, frame, ned, lla, label):
        pt_ned = navpy.lla2ned( lla[0], lla[1], lla[2], self.ref[0], self.ref[1], self.ref[2] )
        rel_ned = [ pt_ned[0] - ned[0], pt_ned[1] - ned[1], pt_ned[2] - ned[2] ]
        dist = math.sqrt(rel_ned[0]*rel_ned[0] + rel_ned[1]*rel_ned[1]
                         + rel_ned[2]*rel_ned[2])
        m2sm = 0.000621371
        dist_sm = dist * m2sm
        if dist_sm <= 15.0:
            scale = 1.0 - dist_sm / 25.0
            if dist_sm <= 7.5:
                label += " (%.1f)" % dist_sm
            # normalize, and draw relative to aircraft ned so that label
            # separation works better
            rel_ned[0] /= dist
            rel_ned[1] /= dist
            rel_ned[2] /= dist
            self.draw_labeled_point(frame,
                                    [ned[0] + rel_ned[0], ned[1] + rel_ned[1],
                                     ned[2] + rel_ned[2]],
                                    label, scale=scale, vert='below')

    def draw_compass_points(self, ned, frame):
        # 30 Ticks
        divs = 12
        pts = []
        for i in range(divs):
            a = (float(i) * 360/float(divs)) * d2r
            n = math.cos(a)
            e = math.sin(a)
            uv1 = self.project_point([ned[0] + n, ned[1] + e, ned[2] - 0.0])
            uv2 = self.project_point([ned[0] + n, ned[1] + e, ned[2] - 0.02])
            if uv1 != None and uv2 != None:
                cv2.line(frame, uv1, uv2, (0,240,0), 1, cv2.CV_AA)

        # North
        uv = self.project_point([ned[0] + 1.0, ned[1] + 0.0, ned[2] - 0.03])
        if uv != None:
            self.draw_label(frame, 'N', uv, 1, self.line_width, vert='above')
        # South
        uv = self.project_point([ned[0] - 1.0, ned[1] + 0.0, ned[2] - 0.03])
        if uv != None:
            self.draw_label(frame, 'S', uv, 1, self.line_width, vert='above')
        # East
        uv = self.project_point([ned[0] + 0.0, ned[1] + 1.0, ned[2] - 0.03])
        if uv != None:
            self.draw_label(frame, 'E', uv, 1, self.line_width, vert='above')
        # West
        uv = self.project_point([ned[0] + 0.0, ned[1] - 1.0, ned[2] - 0.03])
        if uv != None:
            self.draw_label(frame, 'W', uv, 1, self.line_width, vert='above')

    def draw_astro(self, ned, frame, lat_deg, lon_deg, alt_m, timestamp):
        sun_ned, moon_ned = self.compute_sun_moon_ned(lon_deg, lat_deg, alt_m,
                                                      timestamp)
        if sun_ned == None or moon_ned == None:
            return

        # Sun
        self.draw_labeled_point(frame,
                                [ned[0] + sun_ned[0], ned[1] + sun_ned[1],
                                 ned[2] + sun_ned[2]],
                                'Sun')
        # shadow (if sun above horizon)
        if sun_ned[2] < 0.0:
            self.draw_labeled_point(frame,
                                    [ned[0] - sun_ned[0], ned[1] - sun_ned[1],
                                     ned[2] - sun_ned[2]],
                                    'shadow', scale=0.7)
        # Moon
        self.draw_labeled_point(frame,
                                [ned[0] + moon_ned[0], ned[1] + moon_ned[1],
                                 ned[2] + moon_ned[2]],
                                'Moon')

    def draw_airports(self, ned, frame):
        kmsp = [ 44.882000, -93.221802, 256 ]
        self.draw_lla_point(frame, ned, kmsp, 'KMSP')
        ksgs = [ 44.857101, -93.032898, 250 ]
        self.draw_lla_point(frame, ned, ksgs, 'KSGS')
        kstp = [ 44.934502, -93.059998, 215 ]
        self.draw_lla_point(frame, ned, kstp, 'KSTP')
        my52 = [ 44.718601, -93.044098, 281 ]
        self.draw_lla_point(frame, ned, my52, 'MY52')
        kfcm = [ 44.827202, -93.457100, 276 ]
        self.draw_lla_point(frame, ned, kfcm, 'KFCM')
        kane = [ 45.145000, -93.211403, 278 ]
        self.draw_lla_point(frame, ned, kane, 'KANE')
        klvn = [ 44.627899, -93.228104, 293 ]
        self.draw_lla_point(frame, ned, klvn, 'KLVN')
        kmic = [ 45.062000, -93.353897, 265 ]
        self.draw_lla_point(frame, ned, kmic, 'KMIC')
        mn45 = [ 44.566101, -93.132202, 290 ]
        self.draw_lla_point(frame, ned, mn45, 'MN45')
        mn58 = [ 44.697701, -92.864098, 250 ]
        self.draw_lla_point(frame, ned, mn58, 'MN58')
        mn18 = [ 45.187199, -93.130501, 276 ]
        self.draw_lla_point(frame, ned, mn18, 'MN18')

    def draw_nose(self, ned, frame, body2ned):
        vec = transformations.quaternion_transform(body2ned, [1.0, 0.0, 0.0])
        uv = self.project_point([ned[0] + vec[0], ned[1] + vec[1], ned[2]+ vec[2]])
        r1 = int(round(self.render_h / 80))
        r2 = int(round(self.render_h / 40))
        if uv != None:
            cv2.circle(frame, uv, r1, (0,240,0), self.line_width, cv2.CV_AA)
            cv2.circle(frame, uv, r2, (0,240,0), self.line_width, cv2.CV_AA)

    def draw_velocity_vector(self, ned, frame, vel):
        tf = 0.2
        for i in range(3):
            self.vel_filt[i] = (1.0 - tf) * self.vel_filt[i] + tf * vel[i]

        uv = self.project_point([ned[0] + self.vel_filt[0],
                                 ned[1] + self.vel_filt[1],
                                 ned[2] + self.vel_filt[2]])
        if uv != None:
            cv2.circle(frame, uv, 4, (0,240,0), 1, cv2.CV_AA)

    def draw_speed_tape(self, ned, frame, airspeed, ap_speed, flight_mode):
        color = (0,240,0)
        size = 1
        pad = 5 + self.line_width*2
        h, w, d = frame.shape

        # reference point
        cy = int(h * 0.5)
        cx = int(w * 0.2)
        miny = int(h * 0.2)
        maxy = int(h - miny)

        # current airspeed
        label = "%.0f" % airspeed
        lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
        xsize = lsize[0][0] + pad
        ysize = lsize[0][1] + pad
        uv = ( int(cx + ysize*0.7), cy + lsize[0][1] / 2)
        cv2.putText(frame, label, uv, self.font, self.font_size, color, self.line_width, cv2.CV_AA)
        uv1 = (cx, cy)
        uv2 = (cx + int(ysize*0.7),         cy - ysize / 2 )
        uv3 = (cx + int(ysize*0.7) + xsize, cy - ysize / 2 )
        uv4 = (cx + int(ysize*0.7) + xsize, cy + ysize / 2 + 1 )
        uv5 = (cx + int(ysize*0.7),         cy + ysize / 2 + 1)
        cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv2, uv3, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv3, uv4, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv4, uv5, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv5, uv1, color, self.line_width, cv2.CV_AA)

        # speed tics
        spacing = lsize[0][1]
        y = cy - int((0 - airspeed) * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv1 = (cx, y)
        y = cy - int((70 - airspeed) * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv2 = (cx, y)
        cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        for i in range(0, 65, 1):
            offset = int((i - airspeed) * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                uv1 = (cx, cy - offset)
                if i % 5 == 0:
                    uv2 = (cx - 6, cy - offset)
                else:
                    uv2 = (cx - 4, cy - offset)
                cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        for i in range(0, 65, 5):
            offset = int((i - airspeed) * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                label = "%d" % i
                lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
                uv3 = (cx - 8 - lsize[0][0], cy - offset + lsize[0][1] / 2)
                cv2.putText(frame, label, uv3, self.font, self.font_size, color, self.line_width, cv2.CV_AA)

        # speed bug
        offset = int((ap_speed - airspeed) * spacing)
        if flight_mode == 'auto' and cy - offset >= miny and cy - offset <= maxy:
            uv1 = (cx,                  cy - offset)
            uv2 = (cx + int(ysize*0.7), cy - offset - ysize / 2 )
            uv3 = (cx + int(ysize*0.7), cy - offset - ysize )
            uv4 = (cx,                  cy - offset - ysize )
            uv5 = (cx,                  cy - offset + ysize )
            uv6 = (cx + int(ysize*0.7), cy - offset + ysize )
            uv7 = (cx + int(ysize*0.7), cy - offset + ysize / 2 )
            cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv2, uv3, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv3, uv4, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv4, uv5, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv5, uv6, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv6, uv7, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv7, uv1, color, self.line_width, cv2.CV_AA)

    def draw_altitude_tape(self, ned, frame, alt_m, ap_alt, flight_mode):
        color = (0,240,0)
        size = 1
        pad = 5 + self.line_width*2
        h, w, d = frame.shape

        # reference point
        cy = int(h * 0.5)
        cx = int(w * 0.8)
        miny = int(h * 0.2)
        maxy = int(h - miny)

        alt_ft = alt_m / 0.3048
        minrange = int(alt_ft/100)*10 - 30
        maxrange = int(alt_ft/100)*10 + 30

        # current altitude
        label = "%.0f" % (round(alt_ft/10.0) * 10)
        lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
        xsize = lsize[0][0] + pad
        ysize = lsize[0][1] + pad
        uv = ( int(cx - ysize*0.7 - lsize[0][0]), cy + lsize[0][1] / 2)
        cv2.putText(frame, label, uv, self.font, self.font_size, color, self.line_width, cv2.CV_AA)
        uv1 = (cx, cy)
        uv2 = (cx - int(ysize*0.7),         cy - ysize / 2 )
        uv3 = (cx - int(ysize*0.7) - xsize, cy - ysize / 2 )
        uv4 = (cx - int(ysize*0.7) - xsize, cy + ysize / 2 + 1 )
        uv5 = (cx - int(ysize*0.7),         cy + ysize / 2 + 1 )
        cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv2, uv3, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv3, uv4, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv4, uv5, color, self.line_width, cv2.CV_AA)
        cv2.line(frame, uv5, uv1, color, self.line_width, cv2.CV_AA)

        # msl tics
        spacing = lsize[0][1]
        y = cy - int((minrange*10 - alt_ft)/10 * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv1 = (cx, y)
        y = cy - int((maxrange*10 - alt_ft)/10 * spacing)
        if y < miny: y = miny
        if y > maxy: y = maxy
        uv2 = (cx, y)
        cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        for i in range(minrange, maxrange, 1):
            offset = int((i*10 - alt_ft)/10 * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                uv1 = (cx, cy - offset)
                if i % 5 == 0:
                    uv2 = (cx + 6, cy - offset)
                else:
                    uv2 = (cx + 4, cy - offset)
                cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
        for i in range(minrange, maxrange, 5):
            offset = int((i*10 - alt_ft)/10 * spacing)
            if cy - offset >= miny and cy - offset <= maxy:
                label = "%d" % (i*10)
                lsize = cv2.getTextSize(label, self.font, self.font_size, self.line_width)
                uv3 = (cx + 8 , cy - offset + lsize[0][1] / 2)
                cv2.putText(frame, label, uv3, self.font, self.font_size, color, self.line_width, cv2.CV_AA)

        # altitude bug
        offset = int((ap_alt - alt_ft)/10.0 * spacing)
        if flight_mode == 'auto' and cy - offset >= miny and cy - offset <= maxy:
            uv1 = (cx,                  cy - offset)
            uv2 = (cx - int(ysize*0.7), cy - offset - ysize / 2 )
            uv3 = (cx - int(ysize*0.7), cy - offset - ysize )
            uv4 = (cx,                  cy - offset - ysize )
            uv5 = (cx,                  cy - offset + ysize )
            uv6 = (cx - int(ysize*0.7), cy - offset + ysize )
            uv7 = (cx - int(ysize*0.7), cy - offset + ysize / 2 )
            cv2.line(frame, uv1, uv2, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv2, uv3, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv3, uv4, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv4, uv5, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv5, uv6, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv6, uv7, color, self.line_width, cv2.CV_AA)
            cv2.line(frame, uv7, uv1, color, self.line_width, cv2.CV_AA)

