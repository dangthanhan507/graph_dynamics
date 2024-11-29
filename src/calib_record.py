import numpy as np
import cv2
from tqdm import tqdm
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from hardware.teleop_utils import teleop_diagram, CameraTagPublisher
from hardware.cameras import Cameras
import os
#NOTE: run this to record joint positions of robot for calibration

if __name__ == '__main__':
    meshcat = StartMeshcat()
    meshcat.ResetRenderMode()
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
    Ks = cameras.get_intrinsics()
    camera_tag_pub = CameraTagPublisher(cameras, Ks, tag_width=0.056)
    teleop_diag, teleop_scene_graph = teleop_diagram(meshcat, kuka_frame_name="calibration_frame")
    
    teleop_block = builder.AddSystem(teleop_diag)
    cam_block = builder.AddSystem(camera_tag_pub)
    
    builder.Connect(
        teleop_block.GetOutputPort("teleop_pose"),
        cam_block.GetInputPort("tag2kukabase")
    )
    
    builder.ExportOutput(teleop_block.GetOutputPort("iiwa_commanded"), "iiwa_commanded")
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    diagram_context = diagram.GetMutableSubsystemContext(diagram, simulator_context)
    
    simulator.set_target_realtime_rate(1.0)
    
    meshcat.AddButton("Stop Simulation", "Escape")
    meshcat.AddButton("Record Joint", "KeyC")
    current_record_button_ctr = 0
    
    joints = []
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        if meshcat.GetButtonClicks("Record Joint") > current_record_button_ctr:
            current_record_button_ctr += 1
            print("Recording joint position...")
            joint_pos = diagram.GetOutputPort("iiwa_commanded").Eval(diagram_context)
            print(joint_pos)
            joints.append(joint_pos)
            print(f"Recorded joint position: {joint_pos}.")
    meshcat.DeleteButton("Stop Simulation")
    meshcat.DeleteButton("Record Joint")
    
    joints = np.array(joints)
    
    if len(joints) > 0:
        os.makedirs('../calib_data/joints', exist_ok=True)
        np.save('../calib_data/joints/joint_pos.npy', joints)
    print("Done")