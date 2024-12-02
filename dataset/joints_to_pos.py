import sys
sys.path.append('../src')
import numpy as np
from hardware.kuka import joints_to_position
import argparse
import os
#NOTE: convert joints to end-effector particles (just one for now)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--joints", type=str, required=True)
    args = argparser.parse_args()
    folder_path = os.path.dirname(args.joints)
    joints = np.load(args.joints)
    positions = joints_to_position(joints, frame_name='thanos_finger')
    # save to folder
    np.save(os.path.join(folder_path, 'end_effector_positions.npy'), positions)