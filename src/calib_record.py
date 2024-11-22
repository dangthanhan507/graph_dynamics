import numpy as np
from hardware.cameras import Cameras
import cv2

'''
NOTE: Calibrate camera setup on manipulation station using AprilTag

Data Format:
-> cam2tag pose
-> kuka_base2tag pose
'''

if __name__ == '__main__':    
    exposure_time = 10
    cam = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=True)
    cam.start(exposure_time=10)
    obs = cam.get_obs(get_color=True, get_depth=False)
    print(obs.keys())
    print(cam.get_intrinsics().shape)
    
    while 1:
        obs = cam.get_obs(get_color=True, get_depth=False)
        color0 = obs['color_0'][-1]
        color1 = obs['color_1'][-1]
        color2 = obs['color_2'][-1]
        color3 = obs['color_3'][-1]
        
        
        
    cam.stop()
    pass