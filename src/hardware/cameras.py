import cv2
import numpy as np
import time
from hardware.multi_realsense import MultiRealsense, SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import math
import json

class Cameras:
    def __init__(self, WH=[640,480], 
                 capture_fps=15,
                 obs_fps=15, 
                 n_obs_steps=2, 
                 enable_color=True,
                 enable_depth=True,
                 process_depth=True,
                 verbose=False,
                 extrinsic_path=None):
        self.WH = WH
        self.capture_fps = capture_fps
        self.n_obs_steps = n_obs_steps
        self.obs_fps = obs_fps
        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        self.n_fixed_cameras = len(self.serial_numbers)
        print(f"Found {self.n_fixed_cameras} cameras")
        
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.realsense = MultiRealsense(
            serial_numbers=self.serial_numbers,
            shm_manager=self.shm_manager,
            resolution=(self.WH[0], self.WH[1]),
            capture_fps=self.capture_fps,
            enable_color=enable_color,
            enable_depth=enable_depth,
            process_depth=process_depth,
            verbose=verbose)
        self.realsense.set_exposure(exposure=100, gain=60)
        self.realsense.set_white_balance(3800)
        self.last_realsense_data = None
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        
        # NOTE: add in calibration data later
        if extrinsic_path is not None:
            self.extrinsics = np.zeros((self.n_fixed_cameras, 4, 4))
            extrinsics = load_extrinsics(extrinsic_path)
            for i in range(self.n_fixed_cameras):
                self.extrinsics[i] = extrinsics[f'cam{i}']
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready
    
    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()
    
    def start_wait(self):
        self.realsense.start_wait()
    
    def stop_wait(self):
        self.realsense.stop_wait()
        
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    # ========= async env API ===========
    def get_obs(self, get_color=True, get_depth=False) -> dict:
        assert self.is_ready
        
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )
        
        obs = dict()
        dt = 1 / self.obs_fps
        tstamp_list = [x['timestamp'][-1] for x in self.last_realsense_data.values()]
        last_tstamp = np.max(tstamp_list)
        obs_align_tstamps = last_tstamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        
        for camera_idx, value in self.last_realsense_data.items():
            this_tstamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_tstamps:
                is_before_idxs = np.nonzero(this_tstamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            if get_color:
                assert self.enable_color
                obs[f'color_{camera_idx}'] = value['color'][this_idxs]  # BGR
            if get_depth and isinstance(camera_idx, int):
                assert self.enable_depth
                obs[f'depth_{camera_idx}'] = value['depth'][this_idxs] / 1000.0 # mm -> m
        obs['timestamp'] = obs_align_tstamps
        return obs
    
    def get_intrinsics(self):
        return self.realsense.get_intrinsics()
    
    def get_extrinsics(self):
        return self.extrinsics

def depth2pcd(depth, K, rgb=None):
    assert len(depth.shape) == 2
    H,W = depth.shape
    x,y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    # get only non-zero depth
    mask = (depth > 0) & (depth < 0.8)
    points = points[mask, :]
    depth = depth[mask]
    
    points = points * depth[:, None]
    points = points @ np.linalg.inv(K).T
    
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        rgb = rgb[mask, :]
        return points, rgb

    return points

def save_extrinsics(json_dict, filename):
    with open(filename, 'w') as f:
        json.dump(json_dict, f)
def load_extrinsics(filename):
    with open(filename, 'r') as f:
        json_dict = json.load(f)
    # turn all lists into numpy arrays
    for key in json_dict.keys():
        json_dict[key] = np.array(json_dict[key])
    return json_dict