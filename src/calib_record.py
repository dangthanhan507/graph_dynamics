import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from hardware.teleop_utils import teleop_diagram, CameraTagPublisher
from hardware.cameras import Cameras

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
    camera_tag_pub = CameraTagPublisher(cameras, tag_width=0.056)
    teleop_diag,teleop_scene_graph = teleop_diagram(meshcat, kuka_frame_name="calibration_frame")
    
    teleop_block = builder.AddSystem(teleop_diag)
    cam_block = builder.AddSystem(camera_tag_pub)
    
    builder.Connect(
        teleop_block.GetOutputPort("teleop_pose"),
        cam_block.GetInputPort("tag2kukabase")
    )
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    diagram_context = diagram.GetMutableSubsystemContext(diagram, simulator_context)
    
    simulator.set_target_realtime_rate(1.0)
    
    meshcat.AddButton("Stop Simulation", "Escape")
    meshcat.AddButton("Record Image", "KeyC")
    current_record_button_ctr = 0
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        if meshcat.GetButtonClicks("Record Image") > current_record_button_ctr:
            print("Recording Image")
            diagram.ForcedPublish(diagram_context)
            current_record_button_ctr+=1
    meshcat.DeleteButton("Stop Simulation")
    
    print('\n\n')
    print(camera_tag_pub.cameras_datapoints)