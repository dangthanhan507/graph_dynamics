import numpy as np
from pydrake.multibody.inverse_kinematics import DifferentialInverseKinematicsParameters
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from hardware.kuka import create_hardware_diagram_plant
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    ValueProducer,
    AbstractValue,
    MultibodyPlant,
    Multiplexer,
    DifferentialInverseKinematicsIntegrator,
    ConstantValueSource
)
from manipulation.scenarios import AddMultibodyTriad

from hardware.cameras import Cameras
import apriltag
import cv2
from collections import defaultdict

## --- teleop kuka drake API ---
def AddIiwaDifferentialIK(builder, plant, frame=None, trans_vel_limits = 0.03):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0) # nominal position for nullspace projection
    params.set_end_effector_angular_speed_limit(5.0 * np.pi / 180.0)
    params.set_end_effector_translational_velocity_limits(
        [-trans_vel_limits, -trans_vel_limits, -trans_vel_limits], [trans_vel_limits, trans_vel_limits, trans_vel_limits]
    )
    if plant.num_positions() == 3:  # planar iiwa
        iiwa14_velocity_limits = np.array([1.4, 1.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        # These constants are in body frame
        assert (
            frame.name() == "iiwa_link_7"
        ), "Still need to generalize the remaining planar diff IK params for different frames"  # noqa
        params.set_end_effector_velocity_flag(
            [True, False, False, True, False, True]
        )
    else:
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        params.set_joint_centering_gain(0 * np.eye(7))
    if frame is None:
        frame = plant.GetFrameByName("body")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik

class HardwareKukaPose(LeafSystem):
    def __init__(self, hardware_plant: MultibodyPlant, kuka_frame_name = 'iiwa_link_7'):
        LeafSystem.__init__(self)
        self._plant = hardware_plant
        self._plant_context = hardware_plant.CreateDefaultContext()
        self._frame_name = kuka_frame_name
        
        self.DeclareVectorInputPort("kuka_q", 7)
        #make abstract port for kuka pose
        self.DeclareAbstractOutputPort("kuka_pose", lambda: Value(RigidTransform()), self.CalcOutput)
        
    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, q)
        pose = self._plant.GetFrameByName(self._frame_name).CalcPoseInWorld(self._plant_context)
        output.set_value(pose)


def teleop_diagram(meshcat, kuka_frame_name="iiwa_link_7", vel_limits = 0.03):
    meshcat.ResetRenderMode()
    builder = DiagramBuilder()
    
    hardware_diagram, controller_plant, scene_graph = create_hardware_diagram_plant(scenario_filepath='../config/calib_med.yaml', position_only=True, meshcat = meshcat, package_file='../package.xml')
    
    hardware_block = builder.AddSystem(hardware_diagram)
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName(kuka_frame_name),
        trans_vel_limits=vel_limits
    )
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos.position")
    )
    
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos_meshcat.position")
    )
    
    kuka_state = builder.AddSystem(Multiplexer([7,7]))
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.position_measured"),
        kuka_state.get_input_port(0)
    )
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.velocity_estimated"),
        kuka_state.get_input_port(1)
    )
    
    builder.Connect(
        kuka_state.get_output_port(),
        differential_ik.GetInputPort("robot_state"),
    )
    
    use_state = builder.AddSystem(ConstantValueSource(Value(False)))
    builder.Connect(
        use_state.get_output_port(),
        differential_ik.GetInputPort("use_robot_state"),
    )

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    sliders = MeshcatPoseSliders(
        meshcat,
        lower_limit=[-np.pi, -np.pi, -np.pi , -0.6, -0.6, 0.0],
        upper_limit=[ np.pi, np.pi, np.pi, 0.8, 0.9, 2.0],
        step=[5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0, 0.01, 0.01, 0.01],
    )
    
    teleop = builder.AddSystem(sliders)
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    
    # Note: This is using "Cheat Ports". For it to work on hardware, we would
    # need to construct the initial pose from the HardwareStation outputs.
    
    kuka_ee_pose = builder.AddSystem(HardwareKukaPose(controller_plant, kuka_frame_name=kuka_frame_name))
    builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), kuka_ee_pose.get_input_port())
    builder.Connect(kuka_ee_pose.get_output_port(), teleop.get_input_port()) #used for initialization
    AddMultibodyTriad(controller_plant.GetFrameByName("calibration_frame"), scene_graph)

    builder.ExportOutput(kuka_ee_pose.get_output_port(), "teleop_pose")
    builder.ExportOutput(hardware_block.GetOutputPort("iiwa_thanos.position_commanded"), "iiwa_commanded")

    diagram = builder.Build()
    
    return diagram, scene_graph

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

#given 4x4 pose, visualize frame on camera (RGB)
def plotPose(image, pose, length=0.1, thickness=10, K = np.eye(3), dist=np.zeros(5)):
    #NOTE: apriltag x-axis, points out of the tag plane
    rvec = cv2.Rodrigues(pose[:3,:3])[0]
    tvec = pose[:3,3]
    
    axis = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)
    center, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, K, dist)
    center = tuple(center[0].ravel())
    center = (int(center[0]), int(center[1]))
    
    image = cv2.line(image, center, tuple(imgpts[0].ravel()), (0,0,255), thickness=thickness) #red
    image = cv2.line(image, center, tuple(imgpts[1].ravel()), (0,255,0), thickness=thickness)
    image = cv2.line(image, center, tuple(imgpts[2].ravel()), (255,0,0), thickness=thickness)
    return image

def visualize_detections(image, detections):
    for detect in detections:
        image = plotPoint(image, detect.center, CENTER_COLOR)
        image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
        for corner in detect.corners:
            image = plotPoint(image, corner, CORNER_COLOR)
    return image

## --- camera teleop drake API ---
class CameraTagPublisher(LeafSystem):
    def __init__(self, cameras: Cameras, Ks, tag_width: float = 0.056):
        LeafSystem.__init__(self)
        self.tag_width = tag_width
        self.Ks = Ks
        self.cameras = cameras
        self.n_cam = cameras.n_fixed_cameras
        self.detector = apriltag.Detector()
        self.tag2kukabase = RigidTransform()
        self.cameras_datapoints = defaultdict(list)
        self.cam_debug_poses = dict()
        self.obs = None
        
        # take as input Kuka
        self.DeclareAbstractInputPort("tag2kukabase", Value(RigidTransform()))
        
        #get streamed info
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.0, publish=self.SaveObservation)
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.0, publish=self.SaveKukaPose)
        
        #do something with streamed info
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.1, publish=self.VisualizeCameras)
        self.DeclareForcedPublishEvent(self.DetectTagEvent)
        
    def SaveObservation(self, context):
        self.obs = self.cameras.get_obs(get_color=True, get_depth=False)
    def SaveKukaPose(self, context):
        self.tag2kukabase: RigidTransform = self.get_input_port().Eval(context)
        
    def DetectTagEvent(self, context):
        obs = self.obs
        for cam_idx in range(self.n_cam):
            color = obs[f'color_{cam_idx}'][-1]
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detect(gray)
            for detect in detections:
                if detect.tag_id == 0:
                    print(f"Camera {cam_idx} detected tag pose")
                    pt_2d = detect.center
                    pt_3d = self.tag2kukabase.translation()
                    self.cameras_datapoints[f'cam{cam_idx}'].append( (pt_2d,pt_3d) )
                    
                    #get debug pose
                    K = self.Ks[cam_idx]
                    tag2kukabase = self.tag2kukabase.GetAsMatrix4()
                    camera_params = K[0,0],K[1,1],K[0,2],K[1,2]
                    tag2cam, _, _ = self.detector.detection_pose(detect, camera_params=camera_params, tag_size=self.tag_width)
                    cam2tag = np.linalg.inv(tag2cam)
                    cam2kukabase = tag2kukabase @ cam2tag
                    self.cam_debug_poses[f'cam{cam_idx}'] = RigidTransform(cam2kukabase)
                    
        print()
    def VisualizeCameras(self, context):
        assert self.n_cam == 4, "Only works for 4 cameras, doing a 2x2 visual."
        obs = self.obs
        color0 = obs['color_0'][-1]
        detections = self.detector.detect(cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY))
        color0 = visualize_detections(color0, detections)
        
        color1 = obs['color_1'][-1]
        detections = self.detector.detect(cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY))
        color1 = visualize_detections(color1, detections)
        
        color2 = obs['color_2'][-1]
        detections = self.detector.detect(cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY))
        color2 = visualize_detections(color2, detections)
        
        color3 = obs['color_3'][-1]
        detections = self.detector.detect(cv2.cvtColor(color3, cv2.COLOR_BGR2GRAY))
        color3 = visualize_detections(color3, detections)
        
        top = cv2.hconcat([color0, color1])
        bottom = cv2.hconcat([color2, color3])
        fullrgb = cv2.vconcat([top, bottom])
        
        cv2.imshow('rgb', fullrgb)
        if cv2.waitKey(1) == ord('q'):
            return