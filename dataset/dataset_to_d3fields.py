import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import cv2
import shutil

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_folder", type=str, required=True)
    argparser.add_argument("--episode_name", type=str, required=True)
    argparser.add_argument("--output_folder", type=str, required=True)
    args = argparser.parse_args()
    
    
    num_cameras = sum([1 for name in os.listdir(os.path.join(args.data_folder, args.episode_name)) if name.startswith('camera')])
    
    for i in range(num_cameras):
        os.makedirs(f'{args.output_folder}/{args.episode_name}/camera_{i}/', exist_ok=True)
        os.makedirs(f'{args.output_folder}/{args.episode_name}/camera_{i}/color/', exist_ok=True)
        os.makedirs(f'{args.output_folder}/{args.episode_name}/camera_{i}/depth/', exist_ok=True)
    
    # save extrinsic/intrinsic data
    extrinsic_json_filename = os.path.join(args.data_folder, 'camera_extrinsics_robust.json')
    extrinsics_json = json.load(open(extrinsic_json_filename))
    for idx in range(num_cameras):
        extrinsics = np.array(extrinsics_json[f'cam{idx}'])
        np.save(os.path.join(args.output_folder, args.episode_name, f'camera_{idx}', 'camera_extrinsics.npy'), extrinsics)
    intrinsics_json_filename = os.path.join(args.data_folder, 'intrinsics.json')
    intrinsics_json = json.load(open(intrinsics_json_filename))
    for idx in range(num_cameras):
        intrinsics = np.array(intrinsics_json[f'cam{idx}'])
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        camera_params = np.array([fx, fy, cx, cy])
        np.save(os.path.join(args.output_folder, args.episode_name, f'camera_{idx}', 'camera_params.npy'), camera_params)
    
    
    # save camera data to d3fields format
    for idx_cam in range(num_cameras):
        data_folder = os.path.join(args.data_folder, args.episode_name, f'camera_{idx_cam}')
        out_folder  = os.path.join(args.output_folder, args.episode_name, f'camera_{idx_cam}')
        for filename in tqdm(os.listdir(data_folder)):
            if filename.startswith('color'):
                name = filename.split('.')[0] # color_{number}
                number = int(name.split('_')[-1])
                
                color = cv2.imread(os.path.join(data_folder, filename), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(out_folder,f'color/{number}.png'), color)
            elif filename.startswith('depth'):
                name = filename.split('.')[0]
                number = int(name.split('_')[-1])
                
                depth = cv2.imread(os.path.join(data_folder, filename), cv2.IMREAD_ANYDEPTH).astype(np.uint16)
                cv2.imwrite(os.path.join(out_folder,f'depth/{number}.png'), depth)
    
    # save joints
    shutil.copy(os.path.join(args.data_folder, args.episode_name, 'joints.npy'), os.path.join(args.output_folder, args.episode_name, 'joints.npy'))