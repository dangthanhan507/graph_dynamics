#NOTE: goal of this script is to automatically collect data given an object in the middle
import numpy as np
from hardware.cameras import Cameras, depth2pcd
from hardware.kuka import goto_joints_mp
import atexit
import argparse
def getCubePos():
    # get
    pass

def exit_handler(save_folder, joints_recorded):
    np.save(f"{save_folder}/joints_recorded.npy", np.array(joints_recorded))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_folder", type=str, required=True)
    args = argparser.parse_args()
    
    workspace_bbox = np.array([[
        [-0.17, 0.13],
        [0.45, 0.75],
        [0.01, 0.035]
    ]])
    radii_sample = 0.1 # 10cm away from center of object
    cube_length = 0.058 # (m)
    
    home_joint = np.array([ 1.63465136,  0.65955046,  0.07977137, -1.43249199, -0.05626456,  1.05177435, 0.24214364])
    goto_joints_mp(home_joint, endtime=30.0, joint_speed=4.0 * np.pi / 180.0)
    
    joints_recorded = []
    
    
    # atexit.register(exit_handler, args.save_folder, joints_recorded)