import sys
sys.path.append('./')
import numpy as np
import os
import cv2
from hardware.cameras import depth2pcd
class VisualizeEpisode:
    def __init__(self, episode_path):
        self.episode_path = episode_path
        self.length_episode = len(os.listdir(os.path.join(episode_path,'camera_0/color')))
        
        # get all cameras and their extrinsics/intrinsics
        self.num_cameras = len([name for name in os.listdir(episode_path) if name.startswith('camera')])
        
        self.intrinsics = np.zeros((self.num_cameras, 3, 3))
        self.extrinsics = np.zeros((self.num_cameras, 4, 4))
        
        for i in range(self.num_cameras):
            camera_params = np.load(os.path.join(episode_path, f'camera_{i}', 'camera_params.npy'))
            fx, fy, cx, cy = camera_params
            self.intrinsics[i] = np.array([[fx, 0, cx],
                                           [0, fy, cy],
                                           [0, 0, 1]])
            extrinsics = np.load(os.path.join(episode_path, f'camera_{i}', 'camera_extrinsics.npy'))
            self.extrinsics[i] = extrinsics
        
    def __getitem__(self, idx):
        # load depth
        pcds = []
        ptsrgbs = []
        for i in range(self.num_cameras):
            depth = cv2.imread(os.path.join(self.episode_path, f'camera_{i}', 'depth', f'{idx}.png'), cv2.IMREAD_ANYDEPTH) / 1000.0 # in mm still
            color = cv2.imread(os.path.join(self.episode_path, f'camera_{i}', 'color', f'{idx}.png'), cv2.IMREAD_UNCHANGED)
            K = self.intrinsics[i]
            pcd, ptsrgb = depth2pcd(depth, K, rgb=color[:,:,::-1])
            cam2world = np.linalg.inv(self.extrinsics[i])
            R = cam2world[:3, :3]
            t = cam2world[:3,  3]
            pcd = (R @ pcd.T).T + t
            pcds.append(pcd)
            ptsrgbs.append(ptsrgb)
        pcds = np.concatenate(pcds, axis=0)
        ptsrgbs = np.concatenate(ptsrgbs, axis=0)
        return pcds, ptsrgbs
    def __len__(self):
        return self.length_episode