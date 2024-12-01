import sys
sys.path.append('./')
from hardware.cameras import Cameras
import cv2

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
    obs = cam.get_obs(get_color=True, get_depth=True)
    print(obs.keys())
    print(cam.get_intrinsics())
    
    while 1:
        obs = cam.get_obs(get_color=True, get_depth=True)
        topleft = obs['color_0'][-1]
        topright = obs['color_1'][-1]
        bottomleft = obs['color_2'][-1]
        bottomright = obs['color_3'][-1]
        # concatenate the 4 images
        top = cv2.hconcat([topleft, topright])
        bottom = cv2.hconcat([bottomleft, bottomright])
        fullrgb = cv2.vconcat([top, bottom])
        
        topleft = obs['depth_0'][-1]
        topright = obs['depth_1'][-1]
        bottomleft = obs['depth_2'][-1]
        bottomright = obs['depth_3'][-1]
        # concatenate the 4 images
        top = cv2.hconcat([topleft, topright])
        bottom = cv2.hconcat([bottomleft, bottomright])
        fulldepth = cv2.vconcat([top, bottom])
        
        cv2.imshow('rgb', fullrgb)
        cv2.imshow('depth', fulldepth)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop()
    pass