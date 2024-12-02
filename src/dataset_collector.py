import numpy as np
import cv2
from tqdm import tqdm
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from hardware.teleop_utils import teleop_gamepad_diagram
from hardware.cameras import Cameras
from hardware.kuka import goto_joints_mp
from pydrake.all import (
    LeafSystem
)
import argparse
import cv2
import os
import multiprocessing as mp
import time


obs_queue = mp.Queue()
def write_obs(save_folder):
    global obs_queue
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
    for i in range(cameras.n_fixed_cameras):
        os.makedirs(f'{save_folder}/camera_{i}/', exist_ok=True)
    trigger = 5
    index = 0
    while trigger is not None:
        if not obs_queue.empty():
            trigger = obs_queue.get()
            recent_obs = cameras.get_obs(get_color=True, get_depth=True)
            if trigger is None:
                break
            for i in range(cameras.n_fixed_cameras):
                cv2.imwrite(f'{save_folder}/camera_{i}/color_{ "{:04d}".format(index) }.png', recent_obs[f'color_{i}'][-1])
                cv2.imwrite(f'{save_folder}/camera_{i}/depth_{ "{:04d}".format(index) }.png', (recent_obs[f'depth_{i}'][-1]* 1000.0).astype(np.uint16) )
            index += 1
    print("Thread done!")
class Recorder(LeafSystem):
    def __init__(self, save_folder: str):
        LeafSystem.__init__(self)
        self.save_folder = save_folder
        
        #publish at 15Hz
        self.joints_dict = []
        
        self.index = 0
        self.DeclareVectorInputPort("iiwa_pos", 7)
        self.DeclarePeriodicPublishEvent(period_sec=1.0/15.0, offset_sec=0.0, publish=self.publish)
    def start(self):
        self.process_save = mp.Process(target=write_obs, args=(self.save_folder,))
        self.process_save.start()
        time.sleep(5.0)
    def end(self):
        self.process_save.join()
    def publish(self, context):
        global obs_queue
        # make sure context time < 40 seconds
        joints = self.get_input_port().Eval(context)
        
        # save obs
        obs_queue.put(5)
        self.joints_dict.append(joints)
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_folder", type=str, required=True)
    args = argparser.parse_args()
    
    home_joint = np.array([ 1.63465136,  0.65955046,  0.07977137, -1.43249199, -0.05626456,  1.05177435, 0.24214364])
    goto_joints_mp(home_joint, endtime=10.0, joint_speed = 4.0 * np.pi / 180.0, pad_time = 2.0)
    
    meshcat = StartMeshcat()
    meshcat.ResetRenderMode()
    
    builder = DiagramBuilder()
    
    gamepad_diagram, _ = teleop_gamepad_diagram(meshcat, kuka_frame_name="thanos_finger", vel_limits=0.07, scenario_filepath='../config/med.yaml')
    builder.AddSystem(gamepad_diagram)
    recorder = Recorder(args.save_folder)
    recorder.start()
    recorder_block = builder.AddSystem(recorder)
    
    
    builder.Connect(
        gamepad_diagram.GetOutputPort("iiwa_measured"),
        recorder_block.get_input_port()
    )
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    meshcat.AddButton("Stop Simulation", "Escape")
    
    input("Press Enter to Start Recording")
    simulator.AdvanceTo(40.0)
    
    np.save(f'{args.save_folder}/joints.npy', np.array(recorder.joints_dict))
    print("Recording done!")
    
    obs_queue.put(None)
    exit()