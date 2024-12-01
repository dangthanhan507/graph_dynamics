import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from hardware.teleop_utils import teleop_diagram, CameraTagPublisher
from hardware.cameras import Cameras
import os
import argparse
#NOTE: run this to record joint positions of robot for calibration

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--camera", type=bool, default=False)
    argparser.add_argument("--save_file", type=str, required=True)
    args = argparser.parse_args()
    
    meshcat = StartMeshcat()
    meshcat.ResetRenderMode()
    
    if args.camera:
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
        
        Ks = cameras.get_intrinsics()
        camera_tag_pub = CameraTagPublisher(cameras, Ks, tag_width=0.056)
    
    builder = DiagramBuilder()
    teleop_diag, teleop_scene_graph = teleop_diagram(meshcat, kuka_frame_name="calibration_frame")
    
    teleop_block = builder.AddSystem(teleop_diag)
    if args.camera:
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
            # print(f"Recorded joint position: {joint_pos}.")
    meshcat.DeleteButton("Stop Simulation")
    meshcat.DeleteButton("Record Joint")
    
    joints = np.array(joints)
    
    if len(joints) > 0:
        np.save(args.save_file, joints)
    print("Done")