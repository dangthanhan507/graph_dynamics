import numpy as np
from hardware.kuka import goto_joints_mp, curr_pose_mp
import argparse
import apriltag
from tqdm import tqdm
from hardware.cameras import Cameras, save_extrinsics
# given list of joints, follow each joint and take 10 seconds to go to each joint
from collections import defaultdict
import cv2

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, default="../calib_data/joints/joint_pos.npy")
    args = argparser.parse_args()
    
    joints = np.load(args.file) #(N, 7)
    
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=True
    )
    cameras.start(exposure_time=10)
    detector = apriltag.Detector()
    camera_datapoints = defaultdict(list)
    
    print(joints.shape)
    for i in tqdm(range(joints.shape[0])):
        goto_joints_mp(joints[i, :], endtime=30.0, joint_speed=2.0 * np.pi / 180.0)
        pose = curr_pose_mp("../config/calib_med.yaml", "calibration_frame")
        pt3d = pose.translation()
        obs = cameras.get_obs()
        
        for i in range(cameras.n_fixed_cameras):
            color = obs[f'color_{i}'][-1]
            detections = detector.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            for detect in detections:
                if detect.tag_id == 0:
                    pt2d = detect.center
                    camera_datapoints[f'cam{i}'].append((pt2d, pt3d))
    
    Ks = cameras.get_intrinsics()
    camera_json = dict()
    for i in range(cameras.n_fixed_cameras):
        pts2d = np.zeros((len(camera_datapoints[f'cam{i}']), 2))
        pts3d = np.zeros((len(camera_datapoints[f'cam{i}']), 3))
        for j in range(len(camera_datapoints[f'cam{i}'])):
            pt2d, pt3d = camera_datapoints[f'cam{i}'][j]
            pts2d[j, :] = pt2d
            pts3d[j, :] = pt3d
        K = Ks[i]
        ret, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, distCoeffs=np.zeros(5))
        
        rotm = cv2.Rodrigues(rvec)[0]
        H = np.eye(4)
        H[:3, :3] = rotm
        H[:3, 3] = tvec.flatten()
        camera_json[f'cam{i}'] = H.tolist()
        print(np.linalg.inv(H))
    save_extrinsics(camera_json, '../config/camera_extrinsics_robust.json')
    print("Done")