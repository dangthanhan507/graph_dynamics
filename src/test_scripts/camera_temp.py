from hardware.cameras import Cameras
import json
import numpy as np
if __name__ == '__main__':
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=1,
        enable_color=True,
        enable_depth=True,
        process_depth=True,
    )
    cameras.start(exposure_time=10)
    intrinsics = cameras.get_intrinsics()
    # save to json
    intrinsics_json = {}
    for i in range(cameras.n_fixed_cameras):
        intrinsics_json[f'cam{i}'] = intrinsics[i].tolist()
    with open('intrinsics.json', 'w') as f:
        json.dump(intrinsics_json, f)
    cameras.stop(wait=True)