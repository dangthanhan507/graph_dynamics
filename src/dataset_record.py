import numpy as np
from hardware.cameras import Cameras
from hardware.kuka import create_hardware_diagram_plant, goto_joints_mp
from pydrake.all import (
    LeafSystem,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial,
    Simulator
)
import argparse
import cv2
import os
class Recorder(LeafSystem):
    def __init__(self, cameras: Cameras, save_folder: str):
        LeafSystem.__init__(self)
        self.cameras = cameras
        self.save_folder = save_folder
        #publish at 15Hz
        
        for i in range(self.cameras.n_fixed_cameras):
            os.makedirs(f'{self.save_folder}/camera_{i}/', exist_ok=True)
        
        self.index = 0
        self.DeclareVectorInputPort("iiwa_pos", 7)
        self.DeclarePeriodicPublishEvent(period_sec=1.0/15.0, offset_sec=0.0, publish=self.publish)
    def publish(self, context):
        # make sure context time < 40 seconds
        obs = self.cameras.get_obs(get_color=True, get_depth=True)
        joints = self.get_input_port().Eval(context)
        
        # save obs
        for i in range(self.cameras.n_fixed_cameras):
            cv2.imwrite(f'{self.save_folder}/camera_{i}/color_{ "{:04d}".format(self.index) }.png', obs[f'color_{i}'][-1])
            cv2.imwrite(f'{self.save_folder}/camera_{i}/depth_{ "{:04d}".format(self.index) }.png', obs[f'depth_{i}'][-1])
        self.index += 1
        
def record_data(joints: np.ndarray, joint_speed = 4.0 * np.pi / 180.0):
    
    # NOTE: check if initial joint position is the same as current joint position
    j0 = joints[0, :]
    goto_joints_mp(j0, endtime=30.0, joint_speed= 2.0 * np.pi / 180.0, pad_time=2.0)
    input("Press Enter to continue...")
    
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
    builder = DiagramBuilder()
    hardware_diagram, controller_plant, scene_graph = create_hardware_diagram_plant(scenario_filepath='../config/calib_med.yaml', position_only=True, meshcat = None, package_file='../package.xml')
    
    hardware_block = builder.AddSystem(hardware_diagram)
    recorder = Recorder(cameras, '../dataset/rigid/episode_0')
    
    recorder_block = builder.AddSystem(recorder)
    
    #create trajectory from joints and joint_speed
    ts = [0.0]
    for i in range(1,joints.shape[0]):
        max_dq = np.max(np.abs(joints[i, :] - joints[i-1, :]))
        endtime_from_speed = max_dq / joint_speed
        ts.append(ts[-1] + endtime_from_speed)
    ts = np.array(ts)
    
    traj = PiecewisePolynomial.FirstOrderHold(ts, joints.T)
    traj_block = builder.AddSystem(TrajectorySource(traj))
    
    builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), recorder_block.GetInputPort("iiwa_pos"))
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    simulator.AdvanceTo(40.0) #hardcap at 40 seconds

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--load_file", type=str, required=True)
    args = argparser.parse_args()
    
    joints = np.load(args.load_file)
    
    record_data(joints)
    print("Recording done!")