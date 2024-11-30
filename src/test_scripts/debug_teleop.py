import numpy as np
import cv2
from tqdm import tqdm
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from hardware.teleop_utils import teleop_diagram, CameraTagPublisher
from hardware.cameras import Cameras, save_extrinsics

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
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    diagram_context = diagram.GetMutableSubsystemContext(diagram, simulator_context)
    
    simulator.set_target_realtime_rate(1.0)
    
    meshcat.AddButton("Stop Simulation", "Escape")
    meshcat.AddButton("Record Image", "KeyC")
    meshcat.AddButton("Display Counter", "KeyV")
    meshcat.AddButton("Debug Poses", "KeyB")
    meshcat.AddButton("Debug Cam0", "Digit0")
    meshcat.AddButton("Debug Cam1", "Digit1")
    meshcat.AddButton("Debug Cam2", "Digit2")
    meshcat.AddButton("Debug Cam3", "Digit3")
    meshcat.AddButton("Save Debugs", "KeyG")
    current_record_button_ctr = 0
    current_display_button_ctr = 0
    current_debug_button_ctr = 0
    
    current_debug_cam0_ctr = 0
    current_debug_cam1_ctr = 0
    current_debug_cam2_ctr = 0
    current_debug_cam3_ctr = 0
    
    save_debug_json = dict()
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        if meshcat.GetButtonClicks("Record Image") > current_record_button_ctr:
            print("Recording Image")
            diagram.ForcedPublish(diagram_context)
            current_record_button_ctr+=1
        if meshcat.GetButtonClicks("Display Counter") > current_display_button_ctr:
            print("Display Counter:")
            for key in camera_tag_pub.cameras_datapoints.keys():
                print(f"\t{key}: {len(camera_tag_pub.cameras_datapoints[key])}")
            current_display_button_ctr+=1
        if meshcat.GetButtonClicks("Debug Poses") > current_debug_button_ctr:
            print("Debug Poses:")
            for key in camera_tag_pub.cameras_datapoints.keys():
                print(f"\t{key}: {camera_tag_pub.cam_debug_poses[key]}")
            current_debug_button_ctr+=1
        if meshcat.GetButtonClicks("Debug Cam0") > current_debug_cam0_ctr:
            print("Debug Cam0")
            save_debug_json['cam0'] = camera_tag_pub.cam_debug_poses['cam0'].inverse().GetAsMatrix4().tolist()
            current_debug_cam0_ctr+=1
        if meshcat.GetButtonClicks("Debug Cam1") > current_debug_cam1_ctr:
            print("Debug Cam1")
            save_debug_json['cam1'] = camera_tag_pub.cam_debug_poses['cam1'].inverse().GetAsMatrix4().tolist()
            current_debug_cam1_ctr+=1
        if meshcat.GetButtonClicks("Debug Cam2") > current_debug_cam2_ctr:
            print("Debug Cam2")
            save_debug_json['cam2'] = camera_tag_pub.cam_debug_poses['cam2'].inverse().GetAsMatrix4().tolist()
            current_debug_cam2_ctr+=1
        if meshcat.GetButtonClicks("Debug Cam3") > current_debug_cam3_ctr:
            print("Debug Cam3")
            save_debug_json['cam3'] = camera_tag_pub.cam_debug_poses['cam3'].inverse().GetAsMatrix4().tolist()
            current_debug_cam3_ctr+=1
        if meshcat.GetButtonClicks("Save Debugs") > 0:
            print("Saving Debugs")
            save_extrinsics(save_debug_json, '../config/camera_extrinsics.json')
    meshcat.DeleteButton("Stop Simulation")
    meshcat.DeleteButton("Record Image")
    meshcat.DeleteButton("Display Counter")
    meshcat.DeleteButton("Debug Poses")
    meshcat.DeleteButton("Debug Cam0")
    
    # start calibration
    
    # get set of 2d points and 3d points for each camera and calibrate
    camera_data = camera_tag_pub.cameras_datapoints
    camera_extrinsics = np.zeros((cameras.n_fixed_cameras, 4, 4))
    Ks = cameras.get_intrinsics()
    for i in tqdm(len(camera_data)):
        # assert len(camera_data[f"cam{i}"]) > 10, f"Camera {i} needs more data"
        
        pts2d = np.zeros((len(camera_data[f"cam{i}"]), 2))
        pts3d = np.zeros((len(camera_data[f"cam{i}"]), 3))
        for j in range(len(camera_data[f"cam{i}"])):
            pt2d, pt3d = camera_data[f"cam{i}"][j]
            pts2d[j,:] = pt2d
            pts3d[j,:] = pt3d
        K = Ks[i] #intrinsic matrix
        # calibrate
        cam2world_debug = camera_tag_pub.cam_debug_poses[f"cam{i}"].GetAsMatrix4()
        world2cam_debug = np.linalg.inv(cam2world_debug)
        rvec_guess = cv2.Rodrigues(world2cam_debug[:3,:3])[0]
        tvec_guess = world2cam_debug[:3,3].reshape(-1,1)
        
        ret, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, distCoeffs=np.zeros(5), rvec=rvec_guess, tvec=tvec_guess, useExtrinsicGuess=True)
        
        rotm = cv2.Rodrigues(rvec)[0]
        H = np.eye(4)
        H[:3,:3] = rotm
        H[:3,3] = tvec.flatten()
        camera_extrinsics[i] = H
        
    print(H)
    print(np.linalg.inv(H))