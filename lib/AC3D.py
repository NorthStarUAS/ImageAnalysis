import commands
import math
import os.path

def make_textures(project_dir, image_list, resolution=256):
    src_dir = project_dir + '/Images/'
    dst_dir = project_dir + '/Textures/'
    if not os.path.exists(dst_dir):
        print "Notice: creating texture directory =", dst_dir
        os.makedirs(dst_dir)
    for image in image_list:
        src = src_dir + image.name
        dst = dst_dir + image.name
        if not os.path.exists(dst):
            command = "convert -resize %dx%d! %s %s" % ( resolution, resolution,
                                                         src, dst )
            print command
            commands.getstatusoutput( command )
        
def generate(image_list, ref_image=False, project_dir=".", base_name="quick", version=1.0, trans=0.0 ):
    # make the textures if needed
    make_textures(project_dir, image_list, 512)
    
    max_roll = 30.0
    max_pitch = 30.0
    min_agl = 50.0
    min_time = 0.0 # the further into the flight hopefully the better the filter convergence

    ref_lon = None
    ref_lat = None

    # count matching images (starting with 1 to include the reference image)
    match_count = 0
    if ref_image:
        match_count += 1
    match_count += len(image_list)

    # write AC3D header
    name = project_dir
    name += "/"
    name += base_name
    if version:
        name += ("-%02d" % version)
    name += ".ac"
    f = open( name, "w" )
    f.write("AC3Db\n")
    f.write("MATERIAL \"\" rgb 1 1 1  amb 0.6 0.6 0.6  emis 0 0 0  spec 0.5 0.5 0.5  shi 10  trans %.2f\n" % (trans))
    f.write("OBJECT world\n")
    # wrong -- not a real rotation
    #   f.write("rot 1.0 0.0 0.0  0.0 0.0 1.0 0.0 1.0 0.0\n")
    f.write("kids " + str(match_count) + "\n")

    for image in image_list:
        # compute a priority function (higher priority tiles are raised up)
        #priority = (1.0-image.weight) - agl/400.0

        f.write("OBJECT poly\n")
        f.write("name \"rect\"\n")
        f.write("texture \"./Textures/" + image.name + "\"\n")
        f.write("loc 0 0 0\n")

        f.write("numvert %d\n" % len(image.grid_list))
        # output the ac3d polygon grid (note the grid list is in
        # this specific order because that is how we generated it
        # earlier
        pos = 0
        for v in image.grid_list:
            f.write( "%.3f %.3f %.3f\n" % (v[0], v[1], v[2]) )

        steps = int(math.sqrt(len(image.grid_list))) - 1
        f.write("numsurf %d\n" % steps**2)
        dx = 1.0 / float(steps)
        dy = 1.0 / float(steps)
        y = 1.0
        for j in xrange(steps):
            x = 0.0
            for i in xrange(steps):
                c = (j * (steps+1)) + i
                d = ((j+1) * (steps+1)) + i
                f.write("SURF 0x20\n")
                f.write("mat 0\n")
                f.write("refs 4\n")
                f.write("%d %.3f %.3f\n" % (d, x, y-dy))
                f.write("%d %.3f %.3f\n" % (d+1, x+dx, y-dy))
                f.write("%d %.3f %.3f\n" % (c+1, x+dx, y))
                f.write("%d %.3f %.3f\n" % (c, x, y))
                x += dx
            y -= dy
        f.write("kids 0\n")

    if ref_image:
        # reference poly
        f.write("OBJECT poly\n")
        f.write("name \"rect\"\n")
        f.write("texture \"Reference/3drc.png\"\n")
        f.write("loc 0 0 0\n")
        f.write("numvert 4\n")

        f.write(str(gul[0]) + " " + str(gul[1]) + " " + str(gul[2]-15) + "\n")
        f.write(str(gur[0]) + " " + str(gur[1]) + " " + str(gur[2]-15) + "\n")
        f.write(str(glr[0]) + " " + str(glr[1]) + " " + str(glr[2]-15) + "\n")
        f.write(str(gll[0]) + " " + str(gll[1]) + " " + str(gll[2]-15) + "\n")
        f.write("numsurf 1\n")
        f.write("SURF 0x20\n")
        f.write("mat 0\n")
        f.write("refs 4\n")
        f.write("3 0 0\n")
        f.write("2 1 0\n")
        f.write("1 1 1\n")
        f.write("0 0 1\n")
        f.write("kids 0\n")

    f.close()

