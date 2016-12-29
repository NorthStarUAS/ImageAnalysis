import numpy as np

# set camera calibration from some hard coded options
def set_camera_calibration(value):
    if value == 0:
        name = "Mobius 1920x1080 (Curt)"
        K = np.array( [[1362.1,    0.0, 980.8],
                       [   0.0, 1272.8, 601.3],
                       [   0.0,    0.0,   1.0]] )
        dist = [-0.36207197, 0.14627927, -0.00674558, 0.0008926, -0.02635695]
    elif value == 1:
        name = "Mobius 1920x1080 (UMN 003)"
        K = np.array( [[ 1401.21111735,     0.       ,    904.25404757],
                       [    0.        ,  1400.2530882,    490.12157373],
                       [    0.        ,     0.       ,      1.        ]] )
        dist = [-0.39012303,  0.19687255, -0.00069657,  0.00465592, -0.05845262]
    elif value == 2:
        name = "RunCamHD2 1920x1080 (Curt)"
        K = np.array( [[ 971.96149426,   0.        , 957.46750602],
                       [   0.        , 971.67133264, 516.50578382],
                       [   0.        ,   0.        ,   1.        ]] )
        dist = [-0.26910665, 0.10580125, 0.00048417, 0.00000925, -0.02321387]
    elif value == 3:
        name = "RunCamHD2 1920x1440 (Curt)"
        K = np.array( [[ 1296.11187055,     0.        ,   955.43024994],
                       [    0.        ,  1296.01457451,   691.47053988],
                       [    0.        ,     0.        ,     1.        ]] )
        dist = [-0.28250371, 0.14064665, 0.00061846, 0.00014488, -0.05106045]
    else:
        print "unknown camera"
        name = "None"
        K = None
        dist = None
    return name, K, dist
    
